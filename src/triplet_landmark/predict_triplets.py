from __future__ import annotations

import argparse
import csv
import json
import unicodedata
from pathlib import Path

import torch
from PIL import Image

from torch.nn import functional as F

from triplet_landmark.data import make_inference_tensors, resolve_image_path
from triplet_landmark.model import create_model


TRIPLET_COLUMN_CANDIDATES = [
    ("anchor", "positive", "negative"),
    ("anchor_path", "positive_path", "negative_path"),
    ("A1", "A2", "B1"),
    ("a1", "a2", "b1"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write cosine similarities for triplet images.")
    parser.add_argument("--checkpoint", type=Path, nargs="*", default=[])
    parser.add_argument("--triplets", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("outputs/triplet_scores.csv"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--tta", choices=["none", "flip", "five_crop"], default="none")
    parser.add_argument("--embedding-db", type=Path, default=None)
    parser.add_argument("--save-embedding-db", type=Path, default=None)
    parser.add_argument("--anchor-col", default=None)
    parser.add_argument("--positive-col", default=None)
    parser.add_argument("--negative-col", default=None)
    return parser.parse_args()


def resolve_columns(
    fieldnames: list[str],
    anchor_col: str | None,
    positive_col: str | None,
    negative_col: str | None,
) -> tuple[str, str, str]:
    if anchor_col and positive_col and negative_col:
        requested = [anchor_col, positive_col, negative_col]
        missing = [column for column in requested if column not in fieldnames]
        if missing:
            raise ValueError(f"Missing requested triplet columns: {missing}")
        return anchor_col, positive_col, negative_col

    for columns in TRIPLET_COLUMN_CANDIDATES:
        if all(column in fieldnames for column in columns):
            return columns
    raise ValueError(
        "Could not infer triplet columns. Pass --anchor-col, --positive-col, and --negative-col."
    )


def read_triplet_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if path.suffix.lower() == ".json":
        with path.open(encoding="utf-8-sig") as f:
            payload = json.load(f)
        if isinstance(payload, dict) and "triplets" in payload:
            payload = payload["triplets"]
        elif isinstance(payload, dict) and {"anchor", "positive", "negative"} <= set(payload):
            payload = [payload]
        if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
            raise ValueError("Triplet JSON must be a list of objects or an object with a triplets list.")

        fieldnames: list[str] = []
        rows: list[dict[str, str]] = []
        for raw_row in payload:
            row = {str(key): str(value) for key, value in raw_row.items()}
            rows.append(row)
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        return rows, fieldnames

    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = [{key: value for key, value in row.items()} for row in reader]
        return rows, list(reader.fieldnames or [])


def _normalized_strings(value: str) -> list[str]:
    values: list[str] = []
    for form in ("NFC", "NFD", "NFKC", "NFKD"):
        normalized = unicodedata.normalize(form, value)
        if normalized not in values:
            values.append(normalized)
    return values


def resolve_existing_image_path(image_root: Path, value: str) -> Path:
    requested = resolve_image_path(image_root, value)
    if requested.exists():
        return requested

    for candidate_value in _normalized_strings(str(requested)):
        candidate = Path(candidate_value)
        if candidate.exists():
            return candidate

    parent = requested.parent
    if parent.exists():
        requested_names = set(_normalized_strings(requested.name))
        for existing in parent.iterdir():
            if requested_names.intersection(_normalized_strings(existing.name)):
                return existing

    raise FileNotFoundError(
        "Image file not found. "
        f"requested={requested} image_root={image_root} triplet_value={value}"
    )


def load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    label_to_index = checkpoint["label_to_index"]
    args = checkpoint.get("args", {})
    model_name = checkpoint.get("model_name", args.get("model_name", "small_cnn"))
    state_dict = checkpoint["model_state"]
    stored_use_projection = checkpoint.get("use_projection", args.get("use_projection", None))
    has_projection_weights = any(key.startswith("embedding.") for key in state_dict)
    use_projection = bool(stored_use_projection)
    if stored_use_projection is None and model_name != "small_cnn":
        use_projection = has_projection_weights
    pooling = checkpoint.get("pooling", args.get("pooling", None))
    if pooling is None:
        pooling = "gem" if "pool.p" in state_dict else "avg"
    gem_p = float(checkpoint.get("gem_p", args.get("gem_p", 3.0)))
    model = create_model(
        num_classes=len(label_to_index),
        embedding_dim=int(checkpoint["embedding_dim"]),
        model_name=model_name,
        pretrained=False,
        use_projection=use_projection,
        pooling=pooling,
        gem_p=gem_p,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, int(checkpoint["image_size"])


@torch.inference_mode()
def embed_paths(
    paths: list[Path],
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    image_size: int,
    tta: str,
) -> dict[str, torch.Tensor]:
    embeddings: dict[str, torch.Tensor] = {}
    if tta == "none":
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start : start + batch_size]
            images = []
            for path in batch_paths:
                with Image.open(path) as image:
                    images.extend(make_inference_tensors(image, image_size, tta=tta))
            batch = torch.stack(images).to(device)
            batch_embeddings, _ = model(batch)
            for path, embedding in zip(batch_paths, batch_embeddings.cpu()):
                embeddings[str(path)] = embedding
        return embeddings

    for path in paths:
        with Image.open(path) as image:
            tensors = make_inference_tensors(image, image_size, tta=tta)
        batch_embeddings = []
        for start in range(0, len(tensors), batch_size):
            batch = torch.stack(tensors[start : start + batch_size]).to(device)
            embedding, _ = model(batch)
            batch_embeddings.append(embedding.cpu())
        embedding = torch.cat(batch_embeddings, dim=0).mean(dim=0)
        embeddings[str(path)] = F.normalize(embedding, p=2, dim=0)
    return embeddings


def load_embedding_db(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu")
    values = payload["values"]
    embeddings = payload["embeddings"]
    return {str(value): embedding for value, embedding in zip(values, embeddings)}


def save_embedding_db(
    path: Path,
    value_to_embedding: dict[str, torch.Tensor],
    checkpoints: list[Path],
    tta: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = list(value_to_embedding)
    embeddings = torch.stack([value_to_embedding[value].cpu() for value in values])
    torch.save(
        {
            "values": values,
            "embeddings": embeddings,
            "checkpoints": [str(checkpoint) for checkpoint in checkpoints],
            "tta": tta,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    if not args.checkpoint and args.embedding_db is None:
        raise ValueError("Pass at least one --checkpoint, or pass --embedding-db.")

    rows, fieldnames = read_triplet_rows(args.triplets)
    if not rows:
        raise ValueError(f"No triplets found in {args.triplets}")
    if not fieldnames:
        raise ValueError(f"No triplet fields found in {args.triplets}")
    anchor_col, positive_col, negative_col = resolve_columns(
        fieldnames,
        args.anchor_col,
        args.positive_col,
        args.negative_col,
    )

    path_cache: dict[str, Path] = {}

    def cached_path(value: str) -> Path:
        if value not in path_cache:
            path_cache[value] = resolve_existing_image_path(args.image_root, value)
        return path_cache[value]

    unique_values: dict[str, Path] = {}
    for row in rows:
        for column in (anchor_col, positive_col, negative_col):
            unique_values.setdefault(row[column], cached_path(row[column]))

    value_embeddings: dict[str, torch.Tensor] | None = None
    if args.embedding_db is not None:
        value_embeddings = load_embedding_db(args.embedding_db)
        missing = [value for value in unique_values if value not in value_embeddings]
        if missing:
            raise KeyError(f"Embedding DB is missing {len(missing)} images. First missing: {missing[0]}")
    elif len(args.checkpoint) == 1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, image_size = load_model(args.checkpoint[0], device)
        path_embeddings = embed_paths(
            paths=list(unique_values.values()),
            model=model,
            device=device,
            batch_size=args.batch_size,
            image_size=image_size,
            tta=args.tta,
        )
        value_embeddings = {
            value: path_embeddings[str(path)]
            for value, path in unique_values.items()
        }
        if args.save_embedding_db is not None:
            save_embedding_db(args.save_embedding_db, value_embeddings, args.checkpoint, args.tta)

    score_sums = [[0.0, 0.0] for _ in rows]
    if value_embeddings is not None:
        for index, row in enumerate(rows):
            anchor = value_embeddings[row[anchor_col]]
            positive = value_embeddings[row[positive_col]]
            negative = value_embeddings[row[negative_col]]
            score_sums[index][0] = float(torch.dot(anchor, positive))
            score_sums[index][1] = float(torch.dot(anchor, negative))
        score_count = 1
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for checkpoint_path in args.checkpoint:
            model, image_size = load_model(checkpoint_path, device)
            path_embeddings = embed_paths(
                paths=list(unique_values.values()),
                model=model,
                device=device,
                batch_size=args.batch_size,
                image_size=image_size,
                tta=args.tta,
            )
            for index, row in enumerate(rows):
                anchor = path_embeddings[str(unique_values[row[anchor_col]])]
                positive = path_embeddings[str(unique_values[row[positive_col]])]
                negative = path_embeddings[str(unique_values[row[negative_col]])]
                score_sums[index][0] += float(torch.dot(anchor, positive))
                score_sums[index][1] += float(torch.dot(anchor, negative))
        score_count = len(args.checkpoint)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_fields = fieldnames + ["sim_anchor_positive", "sim_anchor_negative"]
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=output_fields)
        writer.writeheader()
        for row, scores in zip(rows, score_sums):
            row["sim_anchor_positive"] = f"{scores[0] / score_count:.8f}"
            row["sim_anchor_negative"] = f"{scores[1] / score_count:.8f}"
            writer.writerow(row)

    print(f"Saved triplet scores: {args.output}")


if __name__ == "__main__":
    main()
