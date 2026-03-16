# src/extract_features.py
"""
Extract features using DINOv2 (if available) or timm ViT-small fallback,
then PCA -> t-SNE and KMeans clustering. Saves tsne.png and cluster0_examples.png.

Seed fixed to 2023090.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

from src.dataset import WildlifeDataset, INV_CLASS_MAP

def detect_device():
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def try_load_dinov2_hub(model_name, device):
    try:
        model = torch.hub.load('facebookresearch/dinov2', model_name)
        model.eval()
        model.to(device)
        print(f"[INFO] Loaded DINOv2 model from torch.hub with name='{model_name}'")
        return model
    except Exception as e:
        print(f"[DEBUG] torch.hub load failed for '{model_name}': {e}")
    return None

def load_timm_fallback(device):
    import timm
    model = timm.create_model('vit_small_patch16_224', pretrained=True)
    if hasattr(model,'head'):
        model.head = torch.nn.Identity()
    elif hasattr(model,'fc'):
        model.fc = torch.nn.Identity()
    else:
        for attr in ['classifier','head','fc']:
            if hasattr(model,attr):
                setattr(model,attr,torch.nn.Identity())
                break
    model.eval()
    model.to(device)
    print("[INFO] Using timm ViT-small fallback")
    return model

def extract_embeddings(model, loader, device):
    all_feats=[]
    all_labels=[]
    all_paths=[]
    model.eval()
    with torch.no_grad():
        for imgs, labels, paths in loader:
            imgs = imgs.to(device)
            feats = model(imgs)
            if isinstance(feats, tuple):
                feats = feats[0]
            if torch.is_tensor(feats):
                feats = feats.detach().cpu().numpy()
            else:
                feats = np.asarray(feats)
            all_feats.append(feats)
            all_labels.extend(labels.numpy().tolist())
            all_paths.extend(paths)
    all_feats = np.concatenate(all_feats, axis=0)
    return all_feats, np.array(all_labels), all_paths

def plot_tsne(z2d, labels, out_path, inv_map):
    plt.figure(figsize=(10,8))
    for cls in np.unique(labels):
        mask = labels==cls
        plt.scatter(z2d[mask,0], z2d[mask,1], label=inv_map[cls], s=10)
    plt.legend(fontsize='small', bbox_to_anchor=(1.05,1))
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print("[INFO] Saved t-SNE plot to", out_path)

def save_cluster_examples(cluster_ids, paths, out_dir, cluster_id=0, n_images=16):
    idxs = np.where(cluster_ids==cluster_id)[0][:n_images]
    fig = plt.figure(figsize=(8,8))
    for i, idx in enumerate(idxs):
        ax = fig.add_subplot(4,4,i+1)
        ax.axis('off')
        img = plt.imread(paths[idx])
        ax.imshow(img)
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"cluster{cluster_id}_examples.png")
    plt.savefig(out_path)
    plt.close()
    print("[INFO] Saved cluster examples to", out_path)

def main(args):
    device = detect_device()
    print("[INFO] Using device:", device)
    tf = T.Compose([T.Resize((224,224)), T.ToTensor(),
                   T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
    ds = WildlifeDataset(csv_file=args.csv, transform=tf)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    model = None
    if args.model:
        model = try_load_dinov2_hub(args.model, device)
    if model is None:
        model = load_timm_fallback(device)
    feats, labels, paths = extract_embeddings(model, loader, device)
    print("[INFO] embeddings shape", feats.shape)
    seed = 2023090
    feats_reduced = feats
    if feats.shape[1] > args.pca_components:
        pca = PCA(n_components=args.pca_components, random_state=seed)
        feats_reduced = pca.fit_transform(feats)
        print("[INFO] PCA reduced to", feats_reduced.shape[1])
    tsne = TSNE(n_components=2, random_state=seed, perplexity=args.perplexity, init='pca')
    z = tsne.fit_transform(feats_reduced)
    os.makedirs(args.out_dir, exist_ok=True)
    tsne_path = os.path.join(args.out_dir, "tsne.png")
    plot_tsne(z, labels, tsne_path, INV_CLASS_MAP)
    k = args.k_clusters
    km = KMeans(n_clusters=k, random_state=seed).fit(feats_reduced)
    cluster_ids = km.labels_
    save_cluster_examples(cluster_ids, paths, args.out_dir, cluster_id=0, n_images=16)
    print("[INFO] Done.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default='data/splits/val.csv')
    parser.add_argument('--out_dir', type=str, default='outputs/features')
    parser.add_argument('--model', type=str, default='dinov2_vits14')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--k_clusters', type=int, default=8)
    parser.add_argument('--perplexity', type=float, default=30.0)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--pca_components', type=int, default=50)
    args = parser.parse_args()
    main(args)