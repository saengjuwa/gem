from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable


def ask_text(label: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"{label}{suffix}: ").strip().strip('"')
    return value or default


def ask_int(label: str, default: int) -> int:
    while True:
        value = ask_text(label, str(default))
        try:
            return int(value)
        except ValueError:
            print("Enter an integer.")


def ask_float_text(label: str, default: str) -> str:
    while True:
        value = ask_text(label, default)
        try:
            float(value)
            return value
        except ValueError:
            print("Enter a number, for example 0.0003 or 1e-5.")


def ask_optional_float_text(label: str, default: str = "") -> str:
    while True:
        value = ask_text(label, default)
        if not value:
            return ""
        try:
            float(value)
            return value
        except ValueError:
            print("Enter a number, for example 0.0003 or 1e-5.")


def ask_yes_no(label: str, default: bool) -> bool:
    default_text = "Y/n" if default else "y/N"
    while True:
        value = input(f"{label} [{default_text}]: ").strip().lower()
        if not value:
            return default
        if value in {"y", "yes"}:
            return True
        if value in {"n", "no"}:
            return False
        print("Enter y or n.")


def run_command(args: list[str]) -> None:
    print("\nCommand:")
    print(subprocess.list2cmdline(args))
    print()
    try:
        subprocess.run(args, cwd=ROOT, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"\nCommand failed with exit code {exc.returncode}.")
    input("\nPress Enter to return to the menu.")


def script_path(name: str) -> str:
    return str(ROOT / "scripts" / name)


def print_training_path_examples() -> None:
    print("Path examples:")
    print("  Label CSV: datasets\\gldv2\\train_10gb_labels.csv")
    print("    Use this after Prepare datasets with the default dataset root.")
    print("  Label CSV: train_10gb_labels.csv")
    print("  Image root: .")
    print("    Use . when the CSV already contains absolute image paths.")
    print("  Image root: D:\\datasets\\gldv2")
    print("    Use this if images are in D:\\datasets\\gldv2\\train_10gb\\")
    print("  Checkpoint: checkpoints\\resnet50_epoch1.pt")


def print_validation_path_examples() -> None:
    print("Path examples:")
    print("  Triplets: datasets\\data\\triplets.json")
    print("    Use this after Prepare datasets with the default dataset root.")
    print("  Triplets: triplets.json")
    print("  Validation image root: datasets\\data\\validation")
    print("    Use this after Prepare datasets with the default dataset root.")
    print("  Validation image root: data\\validation")
    print("    Use this if images are in data\\validation\\00076\\...")
    print("  Validation image root: D:\\datasets\\validation")
    print("    Use this if images are in D:\\datasets\\validation\\00076\\...")
    print("  Checkpoint: checkpoints\\resnet50_epoch1.pt")


def install_requirements() -> None:
    run_command([PYTHON, "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")])


def prepare_data() -> None:
    print("\nPrepare datasets")
    print("This creates these paths under the selected dataset root:")
    print("  gldv2\\train_10gb\\")
    print("  gldv2\\train_10gb_labels.csv")
    print("  data\\validation\\")
    print("Examples:")
    print("  Dataset root: datasets")
    print("  Dataset root: D:\\datasets\\triplet")
    dataset_root = ask_text("Dataset root", "datasets")
    seed = ask_int("GLDv2 train part seed", 420)
    train_parts = ask_int("Train TAR parts, 5 is about 5GB", 5)
    download_train = ask_yes_no("Download/process GLDv2 training data", True)
    download_score = ask_yes_no("Download/unzip scoring data", True)
    keep_tars = ask_yes_no("Keep downloaded train TAR files", False)

    args = [
        PYTHON,
        script_path("prepare_data.py"),
        "--root",
        dataset_root,
        "--seed",
        str(seed),
        "--train-parts",
        str(train_parts),
    ]
    if not download_train:
        args.append("--skip-train")
    if not download_score:
        args.append("--skip-score")
    if keep_tars:
        args.append("--keep-tars")
    run_command(args)


def train() -> None:
    print("\nTrain")
    print_training_path_examples()
    csv_path = ask_text("Label CSV, example datasets\\gldv2\\train_10gb_labels.csv", "datasets\\gldv2\\train_10gb_labels.csv")
    image_root = ask_text("Image root, example . or D:\\datasets\\gldv2", ".")
    model_name = ask_text("Model name", "resnet50")
    pretrained = ask_yes_no("Use pretrained weights", True)
    use_projection = ask_yes_no("Use projection head", False)
    pooling = ask_text("Pooling: avg or gem", "gem")
    gem_p = ask_float_text("GeM p, used only when pooling is gem", "3.0")
    init_checkpoint = ask_text("Init checkpoint, example checkpoints\\resnet50_epoch1.pt, empty to skip", "")
    epochs = ask_int("Epochs", 3)
    batch_size = ask_int("Batch size", 64)
    image_size = ask_int("Image size", 160)
    embedding_dim = ask_int("Embedding dim", 512)
    sampler = ask_text("Sampler: random or balanced", "balanced")
    min_images = ask_int("Min images per label", 2)
    labels_per_batch = ask_int("Labels per batch", 16)
    images_per_label = ask_int("Images per label", 4)
    triplet_weight = ask_float_text("Triplet weight", "0.5")
    triplet_margin = ask_float_text("Triplet margin", "0.2")
    lr = ask_float_text("Head/main LR", "3e-4")
    backbone_lr = ask_optional_float_text("Backbone LR, empty to use one LR", "1e-5")
    output = ask_text("Output checkpoint", f"checkpoints\\{model_name}_windows.pt")

    args = [
        PYTHON,
        script_path("train.py"),
        "--csv",
        csv_path,
        "--image-root",
        image_root,
        "--model-name",
        model_name,
        "--pooling",
        pooling,
        "--gem-p",
        gem_p,
        "--min-images-per-label",
        str(min_images),
        "--sampler",
        sampler,
        "--labels-per-batch",
        str(labels_per_batch),
        "--images-per-label",
        str(images_per_label),
        "--triplet-weight",
        triplet_weight,
        "--triplet-margin",
        triplet_margin,
        "--lr",
        lr,
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--num-workers",
        "2",
        "--image-size",
        str(image_size),
        "--embedding-dim",
        str(embedding_dim),
        "--output",
        output,
    ]
    if pretrained:
        args.append("--pretrained")
    if use_projection:
        args.append("--use-projection")
    if init_checkpoint:
        args.extend(["--init-checkpoint", init_checkpoint])
    if backbone_lr:
        args.extend(["--backbone-lr", backbone_lr])
    run_command(args)


def predict() -> None:
    print("\nPredict triplets from a checkpoint")
    print_validation_path_examples()
    checkpoint = ask_text("Checkpoint, example checkpoints\\resnet50_epoch1.pt", "checkpoints\\resnet50_windows.pt")
    triplets = ask_text("Triplets JSON/CSV, example datasets\\data\\triplets.json", "datasets\\data\\triplets.json")
    image_root = ask_text("Validation image root, example datasets\\data\\validation", "datasets\\data\\validation")
    tta = ask_text("TTA: none, flip, or five_crop", "flip")
    batch_size = ask_int("Batch size", 64)
    output = ask_text("Output score CSV", "outputs\\triplet_scores.csv")

    run_command(
        [
            PYTHON,
            script_path("predict_triplets.py"),
            "--checkpoint",
            checkpoint,
            "--triplets",
            triplets,
            "--image-root",
            image_root,
            "--tta",
            tta,
            "--batch-size",
            str(batch_size),
            "--output",
            output,
        ]
    )


def evaluate() -> None:
    print("\nEvaluate score CSV")
    print("Path example:")
    print("  Score CSV: outputs\\triplet_scores.csv")
    scores = ask_text("Score CSV, example outputs\\triplet_scores.csv", "outputs\\triplet_scores.csv")
    run_command([PYTHON, script_path("evaluate_scores.py"), "--scores", scores])


def build_embedding_db() -> None:
    print("\nBuild embedding DB")
    print_validation_path_examples()
    checkpoint = ask_text("Checkpoint, example checkpoints\\resnet50_epoch1.pt", "checkpoints\\resnet50_windows.pt")
    triplets = ask_text("Triplets JSON/CSV, example datasets\\data\\triplets.json", "datasets\\data\\triplets.json")
    image_root = ask_text("Validation image root, example datasets\\data\\validation", "datasets\\data\\validation")
    tta = ask_text("TTA: none, flip, or five_crop", "flip")
    batch_size = ask_int("Batch size", 64)
    output = ask_text("Output DB", "embedding_db\\validation_flip.pt")

    run_command(
        [
            PYTHON,
            script_path("build_embedding_db.py"),
            "--checkpoint",
            checkpoint,
            "--triplets",
            triplets,
            "--image-root",
            image_root,
            "--tta",
            tta,
            "--batch-size",
            str(batch_size),
            "--output",
            output,
        ]
    )


def predict_from_db() -> None:
    print("\nPredict triplets from an embedding DB")
    print_validation_path_examples()
    print("  Embedding DB: embedding_db\\validation_flip.pt")
    embedding_db = ask_text("Embedding DB, example embedding_db\\validation_flip.pt", "embedding_db\\validation_flip.pt")
    triplets = ask_text("Triplets JSON/CSV, example datasets\\data\\triplets.json", "datasets\\data\\triplets.json")
    image_root = ask_text("Validation image root, example datasets\\data\\validation", "datasets\\data\\validation")
    output = ask_text("Output score CSV, example outputs\\db_scores.csv", "outputs\\db_scores.csv")

    run_command(
        [
            PYTHON,
            script_path("predict_triplets.py"),
            "--embedding-db",
            embedding_db,
            "--triplets",
            triplets,
            "--image-root",
            image_root,
            "--output",
            output,
        ]
    )


def mine_hard_negatives() -> None:
    print("\nMine hard negatives from GLDv2 train images")
    print_training_path_examples()
    checkpoint = ask_text("Checkpoint, example checkpoints\\resnet50_epoch1.pt", "checkpoints\\resnet50_windows.pt")
    csv_path = ask_text("Label CSV, example datasets\\gldv2\\train_10gb_labels.csv", "datasets\\gldv2\\train_10gb_labels.csv")
    image_root = ask_text("Image root, example . or D:\\datasets\\gldv2", ".")
    top_k = ask_int("Top K negatives per anchor", 5)
    limit = ask_text("Limit rows, empty for all", "")
    output = ask_text("Output CSV", "outputs\\hard_negatives.csv")

    args = [
        PYTHON,
        script_path("mine_hard_negatives.py"),
        "--checkpoint",
        checkpoint,
        "--csv",
        csv_path,
        "--image-root",
        image_root,
        "--top-k",
        str(top_k),
        "--output",
        output,
    ]
    if limit:
        args.extend(["--limit", limit])
    run_command(args)


def print_menu() -> None:
    print("\nTriplet Landmark Windows Terminal")
    print("1. Install requirements")
    print("2. Prepare datasets")
    print("3. Train")
    print("4. Predict triplets")
    print("5. Evaluate score CSV")
    print("6. Build embedding DB")
    print("7. Predict from embedding DB")
    print("8. Mine hard negatives")
    print("0. Exit")


def main() -> None:
    parser = argparse.ArgumentParser(description="Windows terminal launcher for this project.")
    parser.add_argument("--list-actions", action="store_true")
    args = parser.parse_args()
    if args.list_actions:
        print_menu()
        return

    actions = {
        "1": install_requirements,
        "2": prepare_data,
        "3": train,
        "4": predict,
        "5": evaluate,
        "6": build_embedding_db,
        "7": predict_from_db,
        "8": mine_hard_negatives,
    }
    while True:
        print_menu()
        choice = input("Select: ").strip()
        if choice == "0":
            return
        action = actions.get(choice)
        if action is None:
            print("Unknown menu item.")
            continue
        action()


if __name__ == "__main__":
    main()
