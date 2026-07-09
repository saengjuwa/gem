from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from triplet_landmark.data import read_label_csv
from triplet_landmark.predict_triplets import embed_paths, load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mine hard negatives from training images only.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--csv", type=Path, default=Path("train_10gb_labels.csv"))
    parser.add_argument("--image-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("outputs/hard_negatives.csv"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive.")

    records, _ = read_label_csv(
        csv_path=args.csv,
        image_root=args.image_root,
        min_images_per_label=1,
        limit=args.limit,
    )
    paths = [record.path for record in records]
    labels = torch.tensor([record.label_index for record in records], dtype=torch.long)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, image_size = load_model(args.checkpoint, device)
    path_embeddings = embed_paths(
        paths=paths,
        model=model,
        device=device,
        batch_size=args.batch_size,
        image_size=image_size,
        tta="none",
    )
    embeddings = torch.stack([path_embeddings[str(path)] for path in paths]).float()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "anchor_path",
                "anchor_label",
                "negative_path",
                "negative_label",
                "similarity",
            ],
        )
        writer.writeheader()

        for start in range(0, len(records), args.chunk_size):
            end = min(start + args.chunk_size, len(records))
            sims = embeddings[start:end] @ embeddings.t()
            for local_index, global_index in enumerate(range(start, end)):
                same_label = labels.eq(labels[global_index])
                negative_count = len(records) - int(same_label.sum().item())
                if negative_count <= 0:
                    continue
                sims[local_index, same_label] = -2.0
                k = min(args.top_k, negative_count)
                values, indices = torch.topk(sims[local_index], k=k)
                for value, negative_index in zip(values.tolist(), indices.tolist()):
                    writer.writerow(
                        {
                            "anchor_path": str(paths[global_index]),
                            "anchor_label": records[global_index].label,
                            "negative_path": str(paths[negative_index]),
                            "negative_label": records[negative_index].label,
                            "similarity": f"{value:.8f}",
                        }
                    )

    print(f"Saved hard negatives: {args.output}")
    print(f"anchors={len(records)} top_k={args.top_k}")


if __name__ == "__main__":
    main()
