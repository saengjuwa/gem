from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from triplet_landmark.data import (
    BalancedBatchSampler,
    ImageTransform,
    LandmarkDataset,
    read_label_csv,
)
from triplet_landmark.model import create_model


def default_csv_path() -> Path:
    colab_csv = Path("/content/gldv2/train_10gb_labels.csv")
    if colab_csv.exists():
        return colab_csv
    return Path("train_10gb_labels.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a landmark embedding baseline.")
    parser.add_argument("--csv", type=Path, default=default_csv_path())
    parser.add_argument("--image-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("checkpoints/baseline.pt"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--model-name", default="small_cnn")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--use-projection", action="store_true")
    parser.add_argument("--pooling", choices=["avg", "gem"], default="avg")
    parser.add_argument("--gem-p", type=float, default=3.0)
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--backbone-lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-images-per-label", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke-test row limit.")
    parser.add_argument("--triplet-weight", type=float, default=0.0)
    parser.add_argument("--triplet-margin", type=float, default=0.2)
    parser.add_argument("--sampler", choices=["random", "balanced"], default="random")
    parser.add_argument("--labels-per-batch", type=int, default=16)
    parser.add_argument("--images-per-label", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=50)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def batch_hard_triplet_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    similarity = embeddings @ embeddings.t()
    distance = 1.0 - similarity
    same_label = labels[:, None].eq(labels[None, :])
    eye = torch.eye(labels.numel(), dtype=torch.bool, device=labels.device)
    positive_mask = same_label & ~eye
    negative_mask = ~same_label

    hardest_positive = distance.masked_fill(~positive_mask, -1.0).max(dim=1).values
    hardest_negative = distance.masked_fill(~negative_mask, 2.0).min(dim=1).values
    valid = positive_mask.any(dim=1) & negative_mask.any(dim=1)
    if not valid.any():
        return embeddings.new_tensor(0.0)
    return F.relu(hardest_positive[valid] - hardest_negative[valid] + margin).mean()


def make_loader(
    records,
    dataset: LandmarkDataset,
    args: argparse.Namespace,
    device: torch.device,
) -> DataLoader:
    if args.sampler == "balanced":
        batch_sampler = BalancedBatchSampler(
            records=records,
            labels_per_batch=args.labels_per_batch,
            images_per_label=args.images_per_label,
            seed=args.seed,
        )
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=len(dataset) > args.batch_size,
    )


def make_optimizer(
    model: torch.nn.Module,
    args: argparse.Namespace,
) -> torch.optim.Optimizer:
    backbone_lr = args.backbone_lr
    if backbone_lr is None:
        return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    backbone_params = []
    head_params = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("backbone."):
            backbone_params.append(parameter)
        else:
            head_params.append(parameter)

    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": backbone_lr})
    if head_params:
        param_groups.append({"params": head_params, "lr": args.lr})

    return torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)


def load_initial_weights(
    model: torch.nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    source_state = checkpoint.get("model_state", checkpoint)
    target_state = model.state_dict()
    compatible_state = {
        key: value
        for key, value in source_state.items()
        if key in target_state and target_state[key].shape == value.shape
    }
    if not compatible_state:
        raise ValueError(f"No compatible weights found in {checkpoint_path}")

    target_state.update(compatible_state)
    model.load_state_dict(target_state)
    skipped = len(source_state) - len(compatible_state)
    print(
        f"Loaded initial weights from {checkpoint_path}: "
        f"{len(compatible_state)} tensors loaded, {skipped} skipped",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    effective_min_images = args.min_images_per_label
    if args.sampler == "balanced" and args.images_per_label > 1:
        effective_min_images = max(effective_min_images, 2)
        if effective_min_images != args.min_images_per_label:
            print(
                "balanced sampler requires at least two real images per label; "
                f"using min_images_per_label={effective_min_images}",
                flush=True,
            )

    records, label_to_index = read_label_csv(
        csv_path=args.csv,
        image_root=args.image_root,
        min_images_per_label=effective_min_images,
        limit=args.limit,
    )
    dataset = LandmarkDataset(records, ImageTransform(args.image_size, train=True))
    loader = make_loader(records, dataset, args, device)

    model = create_model(
        num_classes=len(label_to_index),
        embedding_dim=args.embedding_dim,
        model_name=args.model_name,
        pretrained=args.pretrained,
        use_projection=args.use_projection,
        pooling=args.pooling,
        gem_p=args.gem_p,
    ).to(device)
    if args.init_checkpoint is not None:
        load_initial_weights(model, args.init_checkpoint, device)
    optimizer = make_optimizer(model, args)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(
        f"Training {len(records)} images, {len(label_to_index)} labels, "
        f"model={args.model_name}, pretrained={args.pretrained}, "
        f"use_projection={args.use_projection}, pooling={args.pooling}, "
        f"gem_p={args.gem_p}, sampler={args.sampler}, "
        f"lr={args.lr}, backbone_lr={args.backbone_lr}, "
        f"device={device}, batches={len(loader)}"
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_ce_loss = 0.0
        total_triplet_loss = 0.0
        total_seen = 0
        for step, (images, labels) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                embeddings, logits = model(images)
                ce_loss = F.cross_entropy(logits, labels)
                metric_loss = batch_hard_triplet_loss(
                    embeddings=embeddings,
                    labels=labels,
                    margin=args.triplet_margin,
                )
                loss = ce_loss + args.triplet_weight * metric_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_ce_loss += ce_loss.item() * batch_size
            total_triplet_loss += metric_loss.item() * batch_size
            total_seen += batch_size
            if args.log_every > 0 and step % args.log_every == 0:
                print(
                    f"epoch={epoch} step={step}/{len(loader)} "
                    f"loss={loss.item():.5f} ce={ce_loss.item():.5f} "
                    f"triplet={metric_loss.item():.5f}",
                    flush=True,
                )

        avg_loss = total_loss / max(1, total_seen)
        avg_ce_loss = total_ce_loss / max(1, total_seen)
        avg_triplet_loss = total_triplet_loss / max(1, total_seen)
        print(
            f"epoch={epoch} loss={avg_loss:.5f} "
            f"ce={avg_ce_loss:.5f} triplet={avg_triplet_loss:.5f}",
            flush=True,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "label_to_index": label_to_index,
            "embedding_dim": args.embedding_dim,
            "image_size": args.image_size,
            "model_name": args.model_name,
            "use_projection": args.use_projection,
            "pooling": args.pooling,
            "gem_p": args.gem_p,
            "effective_min_images_per_label": effective_min_images,
            "args": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
        },
        args.output,
    )
    print(f"Saved checkpoint: {args.output}")


if __name__ == "__main__":
    main()
