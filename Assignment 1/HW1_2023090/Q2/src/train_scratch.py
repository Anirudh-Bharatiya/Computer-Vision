# src/train_scratch.py
"""
Train ScratchCNN baseline (Q2.2). No augmentation.
Seed fixed to 2023090.
"""
import argparse, os, time
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from src.dataset import WildlifeDataset, CLASS_MAP, INV_CLASS_MAP
from src.models import ScratchCNN
from src.utils import set_seed, SEED, get_default_num_workers
from src.visualize import plot_confusion_matrix_plt, save_misclassified_grid

try:
    import wandb
    _WANDB_AVAILABLE = True
except Exception:
    wandb = None
    _WANDB_AVAILABLE = False

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    y_true, y_pred = [], []
    for imgs, labels, _ in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        preds = out.argmax(dim=1).cpu().numpy()
        y_pred.extend(preds.tolist())
        y_true.extend(labels.cpu().numpy().tolist())
    avg_loss = running_loss / len(loader.dataset)
    acc = accuracy_score(y_true, y_pred)
    return avg_loss, acc

def eval_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    y_true, y_pred = [], []
    misclassified = []
    with torch.no_grad():
        for imgs, labels, paths in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            out = model(imgs)
            loss = criterion(out, labels)
            running_loss += loss.item() * imgs.size(0)
            preds = out.argmax(dim=1)
            y_pred.extend(preds.cpu().numpy().tolist())
            y_true.extend(labels.cpu().numpy().tolist())
            for i in range(imgs.size(0)):
                if preds[i].item() != labels[i].item():
                    misclassified.append((paths[i], labels[i].item(), int(preds[i].item())))
    avg_loss = running_loss / len(loader.dataset)
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    conf = confusion_matrix(y_true, y_pred)
    return avg_loss, acc, f1_macro, conf, misclassified, y_true, y_pred

def main(args):
    set_seed(SEED)
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    chosen_workers = args.num_workers if args.num_workers is not None else get_default_num_workers(device)
    print(f"[INFO] Device: {device} | num_workers: {chosen_workers}")
    train_tf = T.Compose([T.Resize((224,224)), T.ToTensor(),
                          T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
    val_tf = T.Compose([T.Resize((224,224)), T.ToTensor(),
                        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
    train_ds = WildlifeDataset(csv_file=args.train_csv, transform=train_tf)
    val_ds = WildlifeDataset(csv_file=args.val_csv, transform=val_tf)
    pin_memory = (device.type == 'cuda')
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=chosen_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=chosen_workers, pin_memory=pin_memory)
    model = ScratchCNN(num_classes=len(CLASS_MAP)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    if args.wandb and _WANDB_AVAILABLE:
        wandb.init(project=args.wandb_project, name=args.run_name, config={
            "model":"ScratchCNN","epochs":args.epochs,"lr":args.lr,"batch_size":args.batch_size,
            "seed":SEED,"num_workers":chosen_workers
        })
    os.makedirs(args.out_dir, exist_ok=True)
    best_val_acc = 0.0
    for epoch in range(args.epochs):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1, conf, miscls, y_true, y_pred = eval_epoch(model, val_loader, criterion, device)
        if args.wandb and _WANDB_AVAILABLE:
            wandb.log({"epoch":epoch+1,"train_loss":tr_loss,"train_acc":tr_acc,"val_loss":val_loss,"val_acc":val_acc,"val_f1":val_f1})
        print(f"Epoch {epoch+1}/{args.epochs} train_loss={tr_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} time={(time.time()-t0):.1f}s")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt = os.path.join(args.out_dir, "best_scratch.pth")
            torch.save(model.state_dict(), ckpt)
            print(f"[INFO] Saved best model to {ckpt}")
    print("[INFO] Loading best model for final evaluation...")
    model.load_state_dict(torch.load(os.path.join(args.out_dir, "best_scratch.pth"), map_location=device))
    val_loss, val_acc, val_f1, conf, miscls, y_true, y_pred = eval_epoch(model, val_loader, criterion, device)
    class_names = [INV_CLASS_MAP[i] for i in sorted(INV_CLASS_MAP.keys())]
    fig = plot_confusion_matrix_plt(conf, class_names)
    cm_path = os.path.join(args.out_dir, "confusion_matrix.png")
    fig.savefig(cm_path, bbox_inches='tight')
    if args.wandb and _WANDB_AVAILABLE:
        wandb.log({"confusion_matrix_plt": wandb.Image(cm_path)})
        try:
            cm_table = wandb.plot.confusion_matrix(probs=None, y_true=y_true, preds=y_pred, class_names=class_names)
            wandb.log({"conf_mat_table": cm_table})
        except Exception as e:
            print("[WARN] wandb.plot.confusion_matrix failed:", e)
    if len(miscls)>0:
        mc_path = os.path.join(args.out_dir,"misclassified_grid.png")
        save_misclassified_grid(miscls, out_path=mc_path, per_class=3)
        if args.wandb and _WANDB_AVAILABLE:
            wandb.log({"misclassified_grid": wandb.Image(mc_path)})
    if args.wandb and _WANDB_AVAILABLE:
        wandb.log({"final_val_acc": val_acc, "final_val_f1_macro": val_f1})
        wandb.finish()
    print(f"[FINAL] val_acc={val_acc:.4f}, final_val_f1={val_f1:.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', type=str, default='data/splits/train.csv')
    parser.add_argument('--val_csv', type=str, default='data/splits/val.csv')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--out_dir', type=str, default='outputs')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='HW1_RussianWildlife')
    parser.add_argument('--run_name', type=str, default='scratch_10ep_run')
    parser.add_argument('--num_workers', type=int, default=None, help='If not set, chosen automatically')
    args = parser.parse_args()
    main(args)