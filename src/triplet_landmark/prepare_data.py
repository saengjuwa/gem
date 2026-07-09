from __future__ import annotations

import argparse
import random
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path


TRAIN_METADATA_URL = "https://s3.amazonaws.com/google-landmark/metadata/train.csv"
TRAIN_IMAGE_URL_TEMPLATE = "https://s3.amazonaws.com/google-landmark/train/images_{part}.tar"
SCORE_DATA_FILE_ID = "1-Rt925IS1U-tDfdfOJhB0z3muIyS6gVn"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare GLDv2 train data and scoring data.")
    parser.add_argument("--root", type=Path, required=True, help="Directory where gldv2/ and data/ are created.")
    parser.add_argument("--seed", type=int, default=420, help="Seed used to select GLDv2 train TAR parts.")
    parser.add_argument("--train-parts", type=int, default=5, help="Number of train TAR parts. 5 is about 5GB.")
    parser.add_argument("--skip-train", action="store_true", help="Do not download or process GLDv2 train data.")
    parser.add_argument("--skip-score", action="store_true", help="Do not download or unzip scoring data.")
    parser.add_argument("--keep-tars", action="store_true", help="Keep downloaded train TAR files after extraction.")
    parser.add_argument("--force", action="store_true", help="Redownload metadata, TARs, and score zip if present.")
    parser.add_argument("--score-file-id", default=SCORE_DATA_FILE_ID)
    return parser.parse_args()


def download_with_urllib(url: str, output: Path, force: bool) -> None:
    if output.exists() and not force:
        print(f"already exists, skipping download: {output}")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}")
    urllib.request.urlretrieve(url, output)


def download_with_aria2_or_urllib(url: str, output: Path, force: bool) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not force:
        print(f"already exists, skipping download: {output}")
        return

    aria2c = shutil.which("aria2c")
    if aria2c:
        subprocess.run(
            [
                aria2c,
                "-x",
                "8",
                "-s",
                "8",
                "-c",
                "-d",
                str(output.parent),
                "-o",
                output.name,
                url,
            ],
            check=True,
        )
        return

    print("aria2c not found; using Python downloader. This can be slower.")
    download_with_urllib(url, output, force=force)


def extract_tar(tar_path: Path, output_dir: Path) -> None:
    print(f"extracting {tar_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path) as tar:
        tar.extractall(output_dir)


def download_score_data(root: Path, file_id: str, force: bool) -> None:
    zip_path = root / "data.zip"
    data_dir = root / "data"
    if data_dir.exists() and not force:
        print(f"scoring data already exists, skipping: {data_dir}")
        return

    if not zip_path.exists() or force:
        subprocess.run(
            [sys.executable, "-m", "gdown", file_id, "-O", str(zip_path)],
            check=True,
        )

    print(f"extracting {zip_path}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(root)

    print(f"scoring data: {data_dir}")
    if data_dir.exists():
        for child in sorted(data_dir.iterdir())[:10]:
            print(f"  {child.name}")


def download_metadata(gldv2_dir: Path, force: bool) -> Path:
    metadata_dir = gldv2_dir / "metadata"
    train_csv = metadata_dir / "train.csv"
    download_with_urllib(TRAIN_METADATA_URL, train_csv, force=force)
    return train_csv


def download_train_tars(gldv2_dir: Path, seed: int, train_parts: int, keep_tars: bool, force: bool) -> None:
    if train_parts <= 0 or train_parts > 500:
        raise ValueError("--train-parts must be between 1 and 500.")

    tar_dir = gldv2_dir / "train_tars"
    output_dir = gldv2_dir / "train_10gb"
    tar_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    parts = rng.sample(range(500), train_parts)
    print("chosen parts:", parts)

    for part in parts:
        part_name = f"{part:03d}"
        url = TRAIN_IMAGE_URL_TEMPLATE.format(part=part_name)
        tar_path = tar_dir / f"images_{part_name}.tar"

        download_with_aria2_or_urllib(url, tar_path, force=force)
        extract_tar(tar_path, output_dir)
        if not keep_tars:
            tar_path.unlink(missing_ok=True)


def remove_korean_landmarks(gldv2_dir: Path, train_csv: Path) -> None:
    import pandas as pd
    from datasets import load_dataset

    image_root = gldv2_dir / "train_10gb"
    image_paths = {path.stem: path for path in image_root.rglob("*.jpg")}
    train = pd.read_csv(train_csv, usecols=["id", "landmark_id"])
    train = train[train["id"].isin(image_paths.keys())]

    places = load_dataset("visheratin/google_landmarks_places", split="train").to_pandas()
    kr_countries = {"South Korea", "North Korea"}
    kr_landmark_ids = set(
        places.loc[places["country"].isin(kr_countries), "id"].astype(int)
    )

    remove_df = train[train["landmark_id"].astype(int).isin(kr_landmark_ids)]
    removed = 0
    for image_id in remove_df["id"]:
        path = image_paths.get(image_id)
        if path and path.exists():
            path.unlink()
            removed += 1

    print("downloaded images:", len(image_paths))
    print("korean images removed:", removed)
    print("remaining images:", len(list(image_root.rglob("*.jpg"))))


def write_label_csv(gldv2_dir: Path, train_csv: Path) -> Path:
    import pandas as pd

    image_root = gldv2_dir / "train_10gb"
    output_csv = gldv2_dir / "train_10gb_labels.csv"
    train = pd.read_csv(train_csv, usecols=["id", "landmark_id"])

    files = [{"id": path.stem, "path": str(path)} for path in image_root.rglob("*.jpg")]
    files_df = pd.DataFrame(files)
    labeled = files_df.merge(train, on="id", how="inner")

    landmark_ids = sorted(labeled["landmark_id"].unique())
    id_to_label = {landmark_id: index for index, landmark_id in enumerate(landmark_ids)}
    labeled["label"] = labeled["landmark_id"].map(id_to_label)
    labeled = labeled[["path", "id", "landmark_id", "label"]]
    labeled.to_csv(output_csv, index=False)

    print("images:", len(labeled))
    print("classes:", labeled["landmark_id"].nunique())
    print("saved:", output_csv)
    return output_csv


def prepare_train_data(root: Path, seed: int, train_parts: int, keep_tars: bool, force: bool) -> None:
    gldv2_dir = root / "gldv2"
    train_csv = download_metadata(gldv2_dir, force=force)
    download_train_tars(
        gldv2_dir=gldv2_dir,
        seed=seed,
        train_parts=train_parts,
        keep_tars=keep_tars,
        force=force,
    )
    remove_korean_landmarks(gldv2_dir, train_csv)
    output_csv = write_label_csv(gldv2_dir, train_csv)
    print("train image root:", gldv2_dir / "train_10gb")
    print("train label csv:", output_csv)


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    print("dataset root:", root)
    if not args.skip_score:
        download_score_data(root, args.score_file_id, force=args.force)
    if not args.skip_train:
        prepare_train_data(
            root=root,
            seed=args.seed,
            train_parts=args.train_parts,
            keep_tars=args.keep_tars,
            force=args.force,
        )

    print("done")


if __name__ == "__main__":
    main()
