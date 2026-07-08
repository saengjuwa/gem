from __future__ import annotations

import argparse
from pathlib import Path

import torch

from triplet_landmark.predict_triplets import (
    embed_paths,
    load_model,
    read_triplet_rows,
    resolve_columns,
    resolve_existing_image_path,
    save_embedding_db,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache triplet image embeddings for fast inference.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--triplets", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--tta", choices=["none", "flip", "five_crop"], default="none")
    parser.add_argument("--anchor-col", default=None)
    parser.add_argument("--positive-col", default=None)
    parser.add_argument("--negative-col", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, fieldnames = read_triplet_rows(args.triplets)
    anchor_col, positive_col, negative_col = resolve_columns(
        fieldnames,
        args.anchor_col,
        args.positive_col,
        args.negative_col,
    )

    unique_values: dict[str, Path] = {}
    for row in rows:
        for column in (anchor_col, positive_col, negative_col):
            value = row[column]
            unique_values.setdefault(value, resolve_existing_image_path(args.image_root, value))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, image_size = load_model(args.checkpoint, device)
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
    save_embedding_db(args.output, value_embeddings, [args.checkpoint], args.tta)
    print(f"Saved embedding DB: {args.output}")
    print(f"images={len(value_embeddings)}")


if __name__ == "__main__":
    main()
