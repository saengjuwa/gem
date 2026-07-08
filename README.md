# Triplet Landmark Baseline

This repository now contains a minimal reproducible baseline for the landmark triplet task.

The current baseline trains a small PyTorch image classifier on `train_10gb_labels.csv` and uses the normalized embedding before the classifier head for triplet cosine similarity.

## Data

Expected files:

- `train_10gb_labels.csv`
- `train_10gb/`

The downloader files under `data/` are not used by the training or inference scripts.

## Train

```powershell
python scripts/train.py --csv train_10gb_labels.csv --image-root . --output checkpoints/baseline.pt
```

If the CSV was created in Colab with absolute image paths such as
`/content/gldv2/train_10gb/a/b/c/image.jpg`, no image copy to Drive is needed.
Run training against the Colab runtime files directly:

```bash
python scripts/train.py \
  --csv /content/gldv2/train_10gb_labels.csv \
  --output /content/drive/MyDrive/gldv2_project/checkpoints/baseline.pt
```

When `/content/gldv2/train_10gb_labels.csv` exists, it is also used as the
default CSV path:

```bash
python scripts/train.py --output /content/drive/MyDrive/gldv2_project/checkpoints/baseline.pt
```

For a quick CPU smoke test:

```powershell
python scripts/train.py --limit 128 --epochs 1 --batch-size 8 --num-workers 0 --output checkpoints/smoke.pt
```

## Windows Terminal App

On Windows, run the menu launcher from this project folder:

```powershell
python windows_app.py
```

Or double-click:

```text
run_windows.bat
```

The menu can install requirements, train, predict triplets, evaluate accuracy,
prepare datasets, build an embedding DB, predict from a DB, and mine hard negatives.
Expected local paths after preparing data are:

- `datasets/gldv2/train_10gb_labels.csv`
- `datasets/gldv2/train_10gb/`
- `datasets/data/triplets.json`
- `datasets/data/validation/`

The Windows app remembers the dataset root used in `Prepare datasets`. Train,
predict, embedding DB, and hard-negative menus then use that root as the default
path, while still allowing manual edits in every prompt. The current root is
shown at the top of the menu.

If the data was prepared outside the app, open `Prepare datasets`, enter that
same root, answer `n` to both download questions, and the app will store that
root for later menus.

If `train_10gb_labels.csv` was created in Colab and contains paths such as
`/content/gldv2/train_10gb/...`, the loader maps them back to the local
`train_10gb/` folder when `--image-root .` is used.

Path examples shown in the Windows app:

- Training `Image root`: use `.` when this project folder contains `train_10gb/`.
- Training `Image root`: use `D:\datasets\gldv2` when images are in
  `D:\datasets\gldv2\train_10gb\`.
- Prediction `Validation image root`: use `data\validation` when images are in
  `data\validation\00076\...`.
- Prediction `Validation image root`: use `D:\datasets\validation` when images
  are in `D:\datasets\validation\00076\...`.
- Checkpoint: use a `.pt` file such as `checkpoints\resnet50_epoch1.pt`.

## Prepare Data

The data preparation script downloads both datasets into a user-selected root.
With `--root /content`, it creates the same Colab paths used earlier:

- `/content/gldv2/train_10gb/`
- `/content/gldv2/train_10gb_labels.csv`
- `/content/data/validation/`
- `/content/data/triplets.json`

Colab:

```bash
pip install -q -r requirements.txt
apt-get -qq install aria2
python scripts/prepare_data.py --root /content --seed 420 --train-parts 5
```

Windows:

```powershell
python scripts/prepare_data.py --root D:\datasets\triplet --seed 420 --train-parts 5
```

`--train-parts 5` downloads five GLDv2 train TAR parts, about 5GB before
extraction. The seed controls which parts are selected. The script then removes
South Korea and North Korea landmarks from the GLDv2 training images and writes
`gldv2/train_10gb_labels.csv`.

The scoring data is downloaded with `gdown` from file id
`1-Rt925IS1U-tDfdfOJhB0z3muIyS6gVn` and extracted as `data/`.

Useful variants:

```powershell
python scripts/prepare_data.py --root D:\datasets\triplet --seed 123 --train-parts 5 --skip-score
python scripts/prepare_data.py --root D:\datasets\triplet --skip-train
```

## Stronger Training

External data and pretrained models are allowed by the task rules, but Korean landmark
validation/test images must not be used for training.

Recommended Colab run:

```bash
python scripts/train.py \
  --csv /content/gldv2/train_10gb_labels.csv \
  --model-name convnext_tiny \
  --pretrained \
  --pooling gem \
  --gem-p 3.0 \
  --min-images-per-label 2 \
  --sampler balanced \
  --labels-per-batch 16 \
  --images-per-label 4 \
  --triplet-weight 0.5 \
  --triplet-margin 0.2 \
  --lr 3e-4 \
  --backbone-lr 1e-5 \
  --epochs 5 \
  --batch-size 64 \
  --num-workers 2 \
  --image-size 160 \
  --embedding-dim 512 \
  --output /content/drive/MyDrive/gldv2_project/checkpoints/convnext_tiny_gem.pt
```

This uses:

- `convnext_tiny` pretrained backbone for stronger visual features.
- GeM pooling with `--pooling gem --gem-p 3.0`. GeM can emphasize strong
  landmark-local features more than average pooling, which is often useful for
  image retrieval style embedding tasks.
- Balanced batches so each batch has repeated labels for metric learning.
- Cross-entropy plus batch-hard triplet loss.
- A lower backbone learning rate for ConvNeXt. Training the whole ConvNeXt
  backbone at `3e-4` can collapse embeddings, which produces cosine scores
  near `1.0` for almost every image pair.
- At least two real images per label. The training script now enforces this
  automatically for balanced batches, but passing `--min-images-per-label 2`
  keeps the run command explicit and reproducible.
- No projection head by default for timm backbones. Retrieval uses the pretrained
  backbone feature directly, which avoids early embedding collapse. Use
  `--use-projection` only after this baseline is stable.

Do not pass `/content/data/validation` or Korean landmark metadata to `scripts/train.py`.
Discard old collapsed ConvNeXt checkpoints and retrain with the command above.

## Initialize From A Previous Checkpoint

Use `--init-checkpoint` to start a new training run from a previous model's
weights. This loads compatible tensors only. If the class count changed, the
classifier layer is skipped and the backbone still loads.

```bash
python scripts/train.py \
  --csv /content/gldv2/train_10gb_labels.csv \
  --model-name resnet50 \
  --pretrained \
  --init-checkpoint /content/drive/MyDrive/gldv2_project/checkpoints/resnet50_min2_160.pt \
  --min-images-per-label 2 \
  --sampler balanced \
  --labels-per-batch 16 \
  --images-per-label 4 \
  --triplet-weight 0.5 \
  --triplet-margin 0.2 \
  --epochs 3 \
  --batch-size 64 \
  --num-workers 2 \
  --image-size 160 \
  --embedding-dim 512 \
  --output /content/drive/MyDrive/gldv2_project/checkpoints/resnet50_continued.pt
```

Use the same `--model-name`, `--use-projection`, and `--embedding-dim` as the
source checkpoint unless you intentionally want to load only partial weights.

## Predict Triplets

The default triplet columns are `anchor`, `positive`, and `negative`.
The script also auto-detects `anchor_path,positive_path,negative_path` and `A1,A2,B1`.
Input can be CSV or JSON. For JSON, this shape is supported:

```json
[
  {
    "anchor": "00076/example_a.jpg",
    "positive": "00076/example_b.jpg",
    "negative": "02174/example_c.jpg",
    "anchor_class": "00076",
    "negative_class": "02174"
  }
]
```

```powershell
python scripts/predict_triplets.py --checkpoint checkpoints/baseline.pt --triplets test_triplets.csv --image-root . --output outputs/triplet_scores.csv
```

For the Korean landmark JSON triplet file, set `--image-root` to the folder that contains
the class directories such as `00076/`, `02174/`, and so on:

```powershell
python scripts/predict_triplets.py --checkpoint checkpoints/baseline.pt --triplets triplets.json --image-root path/to/test_images --output outputs/triplet_scores.csv
```

In Colab with `/content/data/validation/00076/...` style folders:

```bash
python scripts/predict_triplets.py \
  --checkpoint /content/drive/MyDrive/gldv2_project/checkpoints/baseline.pt \
  --triplets /content/data/triplets.json \
  --image-root /content/data/validation \
  --output /content/drive/MyDrive/gldv2_project/outputs/triplet_scores.csv
```

Korean filenames are resolved with Unicode normalization fallbacks, so NFC/NFD
filename differences between `triplets.json` and extracted files are handled.

Use TTA at inference:

```bash
python scripts/predict_triplets.py \
  --checkpoint /content/drive/MyDrive/gldv2_project/checkpoints/convnext_tiny_gem.pt \
  --triplets /content/data/triplets.json \
  --image-root /content/data/validation \
  --tta flip \
  --output /content/drive/MyDrive/gldv2_project/outputs/convnext_tta_scores.csv
```

Use an ensemble by passing multiple checkpoints. Similarities are averaged:

```bash
python scripts/predict_triplets.py \
  --checkpoint \
    /content/drive/MyDrive/gldv2_project/checkpoints/convnext_tiny_gem.pt \
    /content/drive/MyDrive/gldv2_project/checkpoints/baseline.pt \
  --triplets /content/data/triplets.json \
  --image-root /content/data/validation \
  --tta flip \
  --output /content/drive/MyDrive/gldv2_project/outputs/ensemble_scores.csv
```

If the released test file uses different column names:

```powershell
python scripts/predict_triplets.py --checkpoint checkpoints/baseline.pt --triplets test_triplets.csv --anchor-col A1 --positive-col A2 --negative-col B1
```

## Output

The prediction CSV keeps the original triplet columns and appends:

- `sim_anchor_positive`
- `sim_anchor_negative`

Accuracy is computed by checking whether `sim_anchor_positive > sim_anchor_negative`.

## Evaluate Accuracy

After prediction, compute triplet accuracy from the score CSV:

```powershell
python scripts/evaluate_scores.py --scores outputs/triplet_scores.csv
```

Colab:

```bash
python scripts/evaluate_scores.py \
  --scores /content/drive/MyDrive/gldv2_project/outputs/triplet_scores.csv
```

## Embedding DB

The embedding DB is a `.pt` file with:

- `values`: original image path strings from the triplet file.
- `embeddings`: L2-normalized image embedding tensor.
- `checkpoints`: checkpoint path used to create the DB.
- `tta`: TTA mode used when extracting embeddings.

It does not train on validation/test images. It only caches inference features so repeated
score generation skips image decoding and model forward passes.

Build a validation embedding DB:

```bash
python scripts/build_embedding_db.py \
  --checkpoint /content/drive/MyDrive/gldv2_project/checkpoints/convnext_tiny_gem.pt \
  --triplets /content/data/triplets.json \
  --image-root /content/data/validation \
  --tta flip \
  --output /content/drive/MyDrive/gldv2_project/embedding_db/validation_convnext_flip.pt
```

Predict from the DB:

```bash
python scripts/predict_triplets.py \
  --embedding-db /content/drive/MyDrive/gldv2_project/embedding_db/validation_convnext_flip.pt \
  --triplets /content/data/triplets.json \
  --image-root /content/data/validation \
  --output /content/drive/MyDrive/gldv2_project/outputs/db_scores.csv
```

## Hard Negatives

Mine hard negatives from training images only:

```bash
python scripts/mine_hard_negatives.py \
  --checkpoint /content/drive/MyDrive/gldv2_project/checkpoints/convnext_tiny_gem.pt \
  --csv /content/gldv2/train_10gb_labels.csv \
  --output /content/drive/MyDrive/gldv2_project/outputs/hard_negatives.csv
```

This file is for experiment analysis and future training improvements. It must be built
from GLDv2 training images, not Korean validation/test images.
