from __future__ import annotations

import csv
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    label: str
    label_index: int


def _label_sort_key(label: str) -> tuple[int, int | str]:
    try:
        return (0, int(label))
    except ValueError:
        return (1, label)


def resolve_image_path(image_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or value.startswith(("/", "\\")):
        if path.exists():
            return path
        parts = path.parts
        for marker in ("train_10gb", "validation"):
            if marker not in parts:
                continue
            marker_index = parts.index(marker)
            candidates = [
                image_root.joinpath(*parts[marker_index:]),
                image_root.joinpath(*parts[marker_index + 1 :]),
            ]
            for candidate in candidates:
                if candidate.exists():
                    return candidate
            if image_root.name == marker:
                return candidates[1]
            return candidates[0]
        return path
    return image_root / path


def read_label_csv(
    csv_path: Path,
    image_root: Path,
    min_images_per_label: int = 1,
    limit: int | None = None,
) -> tuple[list[ImageRecord], dict[str, int]]:
    rows: list[tuple[Path, str]] = []
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"path", "label"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")

        for row in reader:
            image_path = resolve_image_path(image_root, row["path"])
            rows.append((image_path, row["label"]))
            if limit is not None and len(rows) >= limit:
                break

    counts = Counter(label for _, label in rows)
    kept_labels = sorted(
        (label for label, count in counts.items() if count >= min_images_per_label),
        key=_label_sort_key,
    )
    label_to_index = {label: idx for idx, label in enumerate(kept_labels)}

    records = [
        ImageRecord(path=path, label=label, label_index=label_to_index[label])
        for path, label in rows
        if label in label_to_index
    ]
    if not records:
        raise ValueError("No training records left after filtering labels.")
    return records, label_to_index


class ImageTransform:
    def __init__(self, image_size: int, train: bool) -> None:
        self.image_size = image_size
        self.train = train
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB")
        image = self._crop(image)
        image = image.resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)

        return image_to_normalized_tensor(image, self.image_size, self.mean, self.std)

    def _crop(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        side = min(width, height)
        if self.train:
            side = max(1, int(side * random.uniform(0.75, 1.0)))
            left = random.randint(0, max(0, width - side))
            top = random.randint(0, max(0, height - side))
            image = image.crop((left, top, left + side, top + side))
            if random.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            return image

        left = (width - side) // 2
        top = (height - side) // 2
        return image.crop((left, top, left + side, top + side))


def image_to_normalized_tensor(
    image: Image.Image,
    image_size: int,
    mean: torch.Tensor | None = None,
    std: torch.Tensor | None = None,
) -> torch.Tensor:
    if mean is None:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    if std is None:
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    tensor = tensor.view(image_size, image_size, 3).permute(2, 0, 1)
    tensor = tensor.float().div(255.0)
    return (tensor - mean) / std


def make_inference_tensors(
    image: Image.Image,
    image_size: int,
    tta: str = "none",
) -> list[torch.Tensor]:
    image = image.convert("RGB")
    width, height = image.size
    side = min(width, height)
    center_left = (width - side) // 2
    center_top = (height - side) // 2
    boxes = [(center_left, center_top, center_left + side, center_top + side)]

    if tta == "five_crop":
        boxes = [
            (0, 0, side, side),
            (width - side, 0, width, side),
            (0, height - side, side, height),
            (width - side, height - side, width, height),
            (center_left, center_top, center_left + side, center_top + side),
        ]
    elif tta not in {"none", "flip"}:
        raise ValueError(f"Unsupported TTA mode: {tta}")

    tensors: list[torch.Tensor] = []
    for box in boxes:
        crop = image.crop(box).resize((image_size, image_size), Image.Resampling.BICUBIC)
        tensors.append(image_to_normalized_tensor(crop, image_size))
        if tta == "flip":
            flipped = crop.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            tensors.append(image_to_normalized_tensor(flipped, image_size))
    return tensors


class LandmarkDataset(Dataset):
    def __init__(self, records: list[ImageRecord], transform: ImageTransform) -> None:
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]
        with Image.open(record.path) as image:
            tensor = self.transform(image)
        return tensor, torch.tensor(record.label_index, dtype=torch.long)


class BalancedBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        records: list[ImageRecord],
        labels_per_batch: int,
        images_per_label: int,
        batches_per_epoch: int | None = None,
        seed: int = 42,
    ) -> None:
        if labels_per_batch <= 0 or images_per_label <= 0:
            raise ValueError("labels_per_batch and images_per_label must be positive.")

        self.labels_per_batch = labels_per_batch
        self.images_per_label = images_per_label
        self.seed = seed
        self.indices_by_label: dict[int, list[int]] = {}
        for index, record in enumerate(records):
            self.indices_by_label.setdefault(record.label_index, []).append(index)
        if not self.indices_by_label:
            raise ValueError("BalancedBatchSampler received no records.")

        batch_size = labels_per_batch * images_per_label
        self.batches_per_epoch = batches_per_epoch or max(1, len(records) // batch_size)

    def __len__(self) -> int:
        return self.batches_per_epoch

    def __iter__(self):
        rng = random.Random(self.seed)
        labels = list(self.indices_by_label)
        for _ in range(self.batches_per_epoch):
            if len(labels) >= self.labels_per_batch:
                batch_labels = rng.sample(labels, self.labels_per_batch)
            else:
                batch_labels = [rng.choice(labels) for _ in range(self.labels_per_batch)]

            batch: list[int] = []
            for label in batch_labels:
                indices = self.indices_by_label[label]
                if len(indices) >= self.images_per_label:
                    batch.extend(rng.sample(indices, self.images_per_label))
                else:
                    batch.extend(rng.choice(indices) for _ in range(self.images_per_label))
            yield batch
