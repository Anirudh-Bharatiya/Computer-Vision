# scripts/visualize_augmentations.py
"""
Visualize augmented images.
"""
import os
import random
import argparse
import math
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T

SEED = 2023090

def make_aug_pipeline():
    """
    Compose augmentations to apply to PIL images (before ToTensor / Normalize).
    Using >=3 techniques:
      - RandomResizedCrop
      - RandomHorizontalFlip
      - ColorJitter
      - RandomRotation (small)
      - GaussianBlur (optional)
    """
    aug = T.Compose([
        T.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15, hue=0.03),
        # GaussianBlur exists in torchvision >=0.9; adjust kernel size if needed
        T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
    ])
    return aug

def unnormalize_and_to_pil(tensor_img, mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)):
    """
    If you ever have a normalized tensor and want to visualize, unnormalize and convert to PIL.
    Not used in this script since we do augmentations on PIL directly.
    """
    import torchvision.transforms.functional as F
    for t, m, s in zip(tensor_img, mean, std):
        t.mul_(s).add_(m)
    img = tensor_img.clamp(0,1)
    pil = F.to_pil_image(img)
    return pil

def save_augmented_examples(csv_path, out_dir, n_samples=4, n_augs=4, class_idx=None):
    random.seed(SEED)
    df = pd.read_csv(csv_path)
    if class_idx is not None:
        df = df[df['label'] == class_idx]
        if df.empty:
            raise ValueError(f"No examples found for class {class_idx}")
    paths = df['filepath'].values.tolist()
    if len(paths) == 0:
        raise ValueError("No image paths found in CSV")
    # sample without replacement if possible
    n_samples = min(n_samples, len(paths))
    chosen = random.sample(paths, n_samples)
    aug = make_aug_pipeline()

    os.makedirs(out_dir, exist_ok=True)
    saved_records = []  # tuples (orig_path, [aug_paths...])

    for i, p in enumerate(chosen):
        try:
            img = Image.open(p).convert('RGB')
        except Exception as e:
            print(f"[WARN] cannot open {p}: {e}")
            continue
        # Save original resized for comparison
        orig_save = os.path.join(out_dir, f"sample_{i}_orig.png")
        img.resize((224,224)).save(orig_save)
        aug_paths = []
        for j in range(n_augs):
            aimg = aug(img)  # PIL image returned
            aug_save = os.path.join(out_dir, f"sample_{i}_aug_{j}.png")
            aimg.save(aug_save)
            aug_paths.append(aug_save)
        saved_records.append((orig_save, aug_paths))

    # Create and save a montage grid: rows = n_samples, cols = n_augs + 1 (orig + aug)
    rows = len(saved_records)
    cols = (n_augs + 1) if rows > 0 else 0
    if rows == 0:
        raise RuntimeError("No images saved; aborting montage creation")
    fig_w = cols * 3
    fig_h = rows * 3
    fig, axs = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
    if rows == 1:
        axs = [axs]  # make it iterable
    for r, rec in enumerate(saved_records):
        orig, a_list = rec
        row_axs = axs[r] if rows > 1 else axs[0]
        # show original in first column
        im = Image.open(orig).convert('RGB')
        row_axs[0].imshow(im)
        row_axs[0].set_title("orig")
        row_axs[0].axis('off')
        for c, aug_path in enumerate(a_list, start=1):
            im = Image.open(aug_path).convert('RGB')
            row_axs[c].imshow(im)
            row_axs[c].set_title(f"aug{c}")
            row_axs[c].axis('off')
    plt.tight_layout()
    montage_path = os.path.join(out_dir, "augmented_grid.png")
    plt.savefig(montage_path, bbox_inches='tight')
    plt.close(fig)
    print(f"[INFO] Saved augmented images + grid to {out_dir}")
    return saved_records, montage_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default='data/splits/train.csv')
    parser.add_argument('--out_dir', type=str, default='outputs/augments')
    parser.add_argument('--n_samples', type=int, default=4, help='number of distinct source images')
    parser.add_argument('--n_augs', type=int, default=4, help='number of augmented variants per source image')
    parser.add_argument('--class_idx', type=int, default=None, help='optional: restrict to a particular class index')
    parser.add_argument('--seed', type=int, default=SEED)
    args = parser.parse_args()

    # set seeds for reproducibility (random sampling)
    random.seed(args.seed)

    saved_records, montage = save_augmented_examples(args.csv, args.out_dir, n_samples=args.n_samples, n_augs=args.n_augs, class_idx=args.class_idx)
    print("Done. Montage:", montage)