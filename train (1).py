import os
import json
import time
import shutil
import warnings
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import DesertSegDataset, NUM_CLASSES, get_train_augmentation, get_val_augmentation
from model   import build_model
from losses  import CombinedLoss, compute_iou

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

DATA_ROOT = r"C:\Users\nidhi\Documents\COMPUTER ENGINEERING NOTES\hackathon\data\Offroad_Segmentation_Training_Dataset"

OUTPUT_DIR = r"C:\Users\nidhi\Desktop\hackathon_outputs"

TRAIN_DIR = os.path.join(DATA_ROOT, "train")
VAL_DIR   = os.path.join(DATA_ROOT, "val")

IMG_SIZE            = 256
BATCH_SIZE          = 8
EPOCHS              = 25
LR                  = 3e-4
WEIGHT_DECAY        = 1e-4
NUM_WORKERS         = 0
SAVE_EVERY_N_EPOCHS = 5


import shutil
import tempfile
import os
from pathlib import Path

def save_checkpoint(state, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    tmp_path = path.with_suffix('.tmp')
    
    try:
        torch.save(state, tmp_path)
    except RuntimeError as e:
        print(f"  [ERROR] Checkpoint write failed: {e}")
        print(f"  Tip: Check disk space on {path.drive} — checkpoint may be ~200MB+")
        try:
            if tmp_path.exists():
                os.remove(tmp_path)
        except Exception as cleanup_err:
            print(f"  [WARNING] Could not remove temp file {tmp_path}: {cleanup_err}")
        return False
    
    try:
        tmp_path.replace(path)
    except Exception as e:
        print(f"  [WARNING] Could not finalize checkpoint: {e}")
        return False
    
    return True


def load_checkpoint(path: Path, model, optimizer=None, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt.get("epoch", 0), ckpt.get("best_iou", 0.0)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for step, (images, masks) in enumerate(loader):
        images = images.to(device)
        masks  = masks.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, masks)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()

        if (step + 1) % 20 == 0:
            print(f"  step {step+1}/{len(loader)}  loss={loss.item():.4f}")

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_masks = [], []

    for images, masks in loader:
        images = images.to(device)
        masks  = masks.to(device)

        logits = model(images)
        loss   = criterion(logits, masks)

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        all_preds.append(preds.cpu())
        all_masks.append(masks.cpu())

    all_preds = torch.cat(all_preds)
    all_masks = torch.cat(all_masks)
    iou_dict  = compute_iou(all_preds, all_masks, NUM_CLASSES)

    return total_loss / len(loader), iou_dict


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"[train] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("[train] Using CPU")

    ckpt_dir = Path(OUTPUT_DIR) / "checkpoints"
    log_dir  = Path(OUTPUT_DIR) / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[train] Saving outputs to: {OUTPUT_DIR}")

    train_ds = DesertSegDataset(TRAIN_DIR, transform=get_train_augmentation(IMG_SIZE))
    val_ds   = DesertSegDataset(VAL_DIR,   transform=get_val_augmentation(IMG_SIZE))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS)

    model     = build_model(num_classes=NUM_CLASSES).to(device)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    criterion = CombinedLoss(num_classes=NUM_CLASSES).to(device)

    last_ckpt   = ckpt_dir / "last.pth"
    best_ckpt   = ckpt_dir / "best.pth"
    start_epoch = 0
    best_iou    = 0.0

    if last_ckpt.exists():
        try:
            start_epoch, best_iou = load_checkpoint(last_ckpt, model, optimizer, device)
            print(f"[train] ✓ Resumed from epoch {start_epoch}, best mIoU={best_iou:.4f}")
        except Exception as e:
            print(f"[train] Could not load last.pth ({e}), starting fresh.")
            start_epoch, best_iou = 0, 0.0

    history_path = log_dir / "history.json"
    history = []
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)

    for epoch in range(start_epoch, EPOCHS):
        t0 = time.time()
        print(f"\n{'='*55}")
        print(f"  Epoch {epoch+1}/{EPOCHS}")
        print(f"{'='*55}")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, iou_dict = validate(model, val_loader, criterion, device)
        scheduler.step()

        mean_iou = iou_dict["mean_iou"]
        elapsed  = time.time() - t0

        print(f"\n  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"mIoU={mean_iou:.4f}  time={elapsed:.1f}s")

        state = {
            "epoch":     epoch + 1,
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_iou":  best_iou,
        }

        print(f"  Saving last.pth ...", end=" ", flush=True)
        if save_checkpoint(state, last_ckpt):
            print("✓")
        
        if mean_iou > best_iou:
            best_iou = mean_iou
            state["best_iou"] = best_iou
            print(f"  Saving best.pth (mIoU={best_iou:.4f}) ...", end=" ", flush=True)
            if save_checkpoint(state, best_ckpt):
                print("✓")

        if (epoch + 1) % SAVE_EVERY_N_EPOCHS == 0:
            numbered = ckpt_dir / f"epoch_{epoch+1:03d}.pth"
            print(f"  Saving {numbered.name} (backup) ...", end=" ", flush=True)
            if save_checkpoint(state, numbered):
                print("✓")

        history.append({
            "epoch":         epoch + 1,
            "train_loss":    round(train_loss, 4),
            "val_loss":      round(val_loss, 4),
            "mean_iou":      round(mean_iou, 4),
            "per_class_iou": iou_dict["per_class"],
        })
        try:
            with open(history_path, "w") as f:
                json.dump(history, f, indent=2)
        except Exception as e:
            print(f"  [WARNING] Could not save history.json: {e}")

    print(f"\n{'='*55}")
    print(f"  Training complete!  Best mIoU = {best_iou:.4f}")
    print(f"  Checkpoints saved at: {ckpt_dir}")
    print(f"  Run python test.py to generate predictions.")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()


