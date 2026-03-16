# src/visualize.py
"""
Helpers to create confusion matrix images and misclassified image grids.
"""
import os
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import numpy as np

def plot_confusion_matrix_plt(conf_mat, class_names):
    fig, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=class_names, yticklabels=class_names)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    plt.tight_layout()
    return fig

def save_misclassified_grid(misclassified_list, out_path, per_class=3):
    classes = {}
    for path, t, p in misclassified_list:
        classes.setdefault(t, []).append((path,t,p))
    rows = [classes[c][:per_class] for c in sorted(classes.keys())]
    if len(rows) == 0:
        raise ValueError("No misclassified examples provided")
    cols = per_class
    rows_n = len(rows)
    fig, axs = plt.subplots(rows_n, cols, figsize=(cols*3, rows_n*3))
    if rows_n == 1:
        axs = np.expand_dims(axs, 0)
    for r, items in enumerate(rows):
        for c in range(cols):
            ax = axs[r,c]
            ax.axis('off')
            if c < len(items):
                p,t,pred = items[c]
                try:
                    img = Image.open(p).convert('RGB')
                    ax.imshow(img)
                    ax.set_title(f"true:{t}\npred:{pred}")
                except Exception:
                    ax.text(0.5, 0.5, "error", ha='center')
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return fig