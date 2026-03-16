# Image Classification (Russian Wildlife)

This repository contains code to reproduce **Q2 (Image Classification)** for the assignment.

Run the three required experiments:
- Scratch CNN baseline
- ResNet-18 finetune **without augmentation**
- ResNet-18 finetune **with augmentation**

Also includes **DINOv2 / timm feature extraction with PCA → t-SNE + KMeans clustering**.

Seed used throughout: `2023090`.

---

## Run commands

### 1. Create dataset splits

```bash
python -m src.dataset \
  --root path/to/wildlife_dataset \
  --make-splits \
  --out_csv_dir data/splits \
  --seed 2023090
```

---

### 2. Plot class distributions (Q2.1.c)

```bash
python scripts/plot_class_distribution.py
```

This generates:

```
outputs/plots/train_class_dist.png
outputs/plots/val_class_dist.png
```

---

### 3. Scratch CNN baseline (10 epochs) — Q2.2

```bash
python -m src.train_scratch \
  --train_csv data/splits/train.csv \
  --val_csv data/splits/val.csv \
  --epochs 10 \
  --batch_size 64 \
  --out_dir outputs/scratch_baseline \
  --wandb \
  --wandb_project "HW1_RussianWildlife" \
  --run_name "scratch_10ep_run"
```

---

### 4. ResNet-18 finetune **without augmentation** — Q2.3

```bash
python -m src.train_finetune \
  --train_csv data/splits/train.csv \
  --val_csv data/splits/val.csv \
  --epochs 10 \
  --batch_size 32 \
  --out_dir outputs/resnet_no_aug \
  --wandb \
  --wandb_project "HW1_RussianWildlife" \
  --run_name "resnet18_no_aug"
```

---

### 5. ResNet-18 finetune **with augmentation** — Q2.4

```bash
python -m src.train_finetune \
  --train_csv data/splits/train.csv \
  --val_csv data/splits/val.csv \
  --epochs 10 \
  --batch_size 32 \
  --out_dir outputs/resnet_with_aug \
  --wandb \
  --wandb_project "HW1_RussianWildlife" \
  --run_name "resnet18_with_aug" \
  --augment
```

---

### 6. Feature extraction + PCA → t-SNE + KMeans (Q3.2.d)

```bash
python -m src.extract_features \
  --csv data/splits/val.csv \
  --out_dir outputs/features \
  --model dinov2_vits14 \
  --k_clusters 8
```