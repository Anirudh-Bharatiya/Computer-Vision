# src/train_finetune.py
"""
Finetune ResNet-18 (Q2.3 & Q2.4). Use --augment for data augmentation experiments.
Seed fixed to 2023090.
"""
import argparse, os, time
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from src.models import make_resnet18
from src.dataset import WildlifeDataset, CLASS_MAP, INV_CLASS_MAP
from src.utils import set_seed, SEED, get_default_num_workers
from src.visualize import plot_confusion_matrix_plt, save_misclassified_grid

try:
    import wandb
    _WANDB_AVAILABLE = True
except Exception:
    wandb = None
    _WANDB_AVAILABLE = False

def eval_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    y_true, y_pred = [], []
    misclassified=[]
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
    if args.augment:
        train_tf = T.Compose([
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.2,0.2,0.2,0.1),
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
    else:
        train_tf = T.Compose([T.Resize((224,224)), T.ToTensor(),
                              T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
    val_tf = T.Compose([T.Resize((224,224)), T.ToTensor(),
                        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
    train_ds = WildlifeDataset(csv_file=args.train_csv, transform=train_tf)
    val_ds = WildlifeDataset(csv_file=args.val_csv, transform=val_tf)
    pin_memory = (device.type == 'cuda')
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=chosen_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=chosen_workers, pin_memory=pin_memory)
    model = make_resnet18(num_classes=len(CLASS_MAP), pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    if args.wandb and _WANDB_AVAILABLE:
        wandb.init(project=args.wandb_project, name=args.run_name, config={
            "model":"resnet18_finetune","epochs":args.epochs,"lr":args.lr,"augment":args.augment,"seed":SEED,"num_workers":chosen_workers
        })
    os.makedirs(args.out_dir, exist_ok=True)
    best_acc = 0.0
    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        y_true, y_pred = [], []
        for imgs, labels, _ in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            y_pred.extend(out.argmax(dim=1).cpu().numpy().tolist())
            y_true.extend(labels.cpu().numpy().tolist())
        train_loss = total_loss / len(train_loader.dataset)
        train_acc = accuracy_score(y_true, y_pred)
        v_loss, v_acc, v_f1, conf, miscls, vy_true, vy_pred = eval_epoch(model, val_loader, criterion, device)
        if args.wandb and _WANDB_AVAILABLE:
            wandb.log({"epoch":epoch+1,"train_loss":train_loss,"train_acc":train_acc,"val_loss":v_loss,"val_acc":v_acc,"val_f1":v_f1})
        print(f"Epoch {epoch+1}/{args.epochs} train_loss={train_loss:.4f} val_loss={v_loss:.4f} val_acc={v_acc:.4f} time={(time.time()-t0):.1f}s")
        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(model.state_dict(), os.path.join(args.out_dir, 'best_resnet18.pth'))
            print(f"[INFO] Saved best model to {os.path.join(args.out_dir, 'best_resnet18.pth')}")
    print("[INFO] Loading best model for final evaluation...")
    model.load_state_dict(torch.load(os.path.join(args.out_dir, 'best_resnet18.pth'), map_location=device))
    val_loss, val_acc, val_f1, conf, miscls, y_true, y_pred = eval_epoch(model, val_loader, criterion, device)
    class_names = [INV_CLASS_MAP[i] for i in sorted(INV_CLASS_MAP.keys())]
    fig = plot_confusion_matrix_plt(conf, class_names)
    cm_path = os.path.join(args.out_dir, "confusion_matrix_resnet.png")
    fig.savefig(cm_path, bbox_inches='tight')
    if args.wandb and _WANDB_AVAILABLE:
        wandb.log({"confusion_matrix_plt":wandb.Image(cm_path)})
        try:
            cm_table = wandb.plot.confusion_matrix(probs=None, y_true=y_true, preds=y_pred, class_names=class_names)
            wandb.log({"conf_mat_table": cm_table})
        except Exception as e:
            print("[WARN] wandb.plot.confusion_matrix failed:", e)
    if len(miscls)>0:
        mc_path = os.path.join(args.out_dir, f"misclassified_grid_resnet_{'aug' if args.augment else 'base'}.png")
        save_misclassified_grid(miscls, out_path=mc_path, per_class=3)
        if args.wandb and _WANDB_AVAILABLE:
            wandb.log({"misclassified_grid": wandb.Image(mc_path)})
    if args.wandb and _WANDB_AVAILABLE:
        wandb.log({"final_val_acc": val_acc, "final_val_f1_macro": val_f1})
        wandb.finish()
    print(f"[FINAL] val_acc={val_acc:.4f}, val_f1={val_f1:.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', type=str, default='data/splits/train.csv')
    parser.add_argument('--val_csv', type=str, default='data/splits/val.csv')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--out_dir', type=str, default='outputs')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='HW1_RussianWildlife')
    parser.add_argument('--run_name', type=str, default='resnet18_finetune')
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--num_workers', type=int, default=None)
    args = parser.parse_args()
    main(args)