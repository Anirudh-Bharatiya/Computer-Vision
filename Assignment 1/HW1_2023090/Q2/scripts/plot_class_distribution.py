# scripts/plot_class_distribution.py
import os
import pandas as pd
import matplotlib.pyplot as plt
from src.dataset import INV_CLASS_MAP

def plot_from_csv(csv_path, out_path, title=""):
    df = pd.read_csv(csv_path)
    counts = df['label'].value_counts().sort_index()
    labels = [INV_CLASS_MAP[i] for i in counts.index]
    plt.figure(figsize=(10,6))
    plt.bar(labels, counts.values)
    plt.xticks(rotation=45, ha='right')
    plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    print("Saved", out_path)

if __name__ == "__main__":
    os.makedirs("outputs/plots", exist_ok=True)
    plot_from_csv("data/splits/train.csv", "outputs/plots/train_class_dist.png", "Train class distribution")
    plot_from_csv("data/splits/val.csv", "outputs/plots/val_class_dist.png", "Val class distribution")