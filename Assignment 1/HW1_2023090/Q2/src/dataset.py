# src/dataset.py
"""
Dataset & split helper for the Russian Wildlife dataset.

Creates stratified splits (0.8/0.2) and provides a PyTorch Dataset wrapper.
"""
import os
from glob import glob
import pandas as pd
import argparse
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset
from PIL import Image

CLASS_MAP = {
    'amur leopard': 0,
    'amur tiger': 1,
    'birds': 2,
    'black bear': 3,
    'brown bear': 4,
    'dog': 5,
    'roe deer': 6,
    'sika deer': 7,
    'wild boar': 8,
    'people': 9
}
INV_CLASS_MAP = {v: k for k, v in CLASS_MAP.items()}

def make_df_from_folder(root: str):
    rows = []
    for cls_name, cls_id in CLASS_MAP.items():
        variants = [
            os.path.join(root, cls_name),
            os.path.join(root, cls_name.replace(' ', '_')),
            os.path.join(root, cls_name.replace(' ', '-'))
        ]
        for d in variants:
            if os.path.isdir(d):
                for p in glob(os.path.join(d, '*')):
                    if p.lower().endswith(('.jpg', '.jpeg', '.png')):
                        rows.append({'filepath': p, 'label': cls_id})
                break
    return pd.DataFrame(rows)

def stratified_split(df, out_dir, seed, ratio=0.8):
    sss = StratifiedShuffleSplit(n_splits=1, train_size=ratio, random_state=int(seed))
    X = df['filepath'].values
    y = df['label'].values
    train_idx, val_idx = next(sss.split(X, y))
    tr = df.iloc[train_idx].reset_index(drop=True)
    vl = df.iloc[val_idx].reset_index(drop=True)
    os.makedirs(out_dir, exist_ok=True)
    tr.to_csv(os.path.join(out_dir, 'train.csv'), index=False)
    vl.to_csv(os.path.join(out_dir, 'val.csv'), index=False)
    print(f"[INFO] Saved splits to {out_dir} (train={len(tr)}, val={len(vl)})")
    return tr, vl

class WildlifeDataset(Dataset):
    def __init__(self, csv_file=None, root=None, transform=None):
        if csv_file is None and root is None:
            raise ValueError("Provide csv_file or root")
        if csv_file is not None:
            df = pd.read_csv(csv_file)
        else:
            df = make_df_from_folder(root)
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row['filepath']
        label = int(row['label'])
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label, path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--out_csv_dir', type=str, default='data/splits')
    parser.add_argument('--seed', type=int, default=2023090)
    parser.add_argument('--make-splits', action='store_true')
    args = parser.parse_args()
    if args.make_splits:
        if os.path.isdir(args.root):
            df = make_df_from_folder(args.root)
        else:
            df = pd.read_csv(args.root)
        print(f"[INFO] Found {len(df)} rows. Creating stratified splits...")
        stratified_split(df, args.out_csv_dir, args.seed)
    else:
        print("[INFO] Use --make-splits to create CSV splits.")