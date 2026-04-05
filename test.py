import os
import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from dataset import DesertTestDataset, NUM_CLASSES, CLASS_COLORS
from model   import build_model

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

TEST_DIR = r"C:\Users\nidhi\Desktop\Bonus"
CHECKPOINT = "C:\\Users\\nidhi\\Desktop\\hackathon_outputs\\checkpoints\\last.pth"
OUTPUT_DIR = r"C:\Users\nidhi\Desktop\hackathon_outputs"
IMG_SIZE    = 256
BATCH_SIZE  = 8
NUM_WORKERS = 0


def colorize_mask(mask_np: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*mask_np.shape, 3), dtype=np.uint8)
    for cls_idx, color in enumerate(CLASS_COLORS):
        rgb[mask_np == cls_idx] = color
    return rgb


def save_predictions(pred_mask, original_img, stem, out_dir):
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(exist_ok=True)
    Image.fromarray(pred_mask.astype(np.uint8)).save(raw_dir / f"{stem}.png")

    color_dir = out_dir / "color"
    color_dir.mkdir(exist_ok=True)
    color_img = colorize_mask(pred_mask)
    Image.fromarray(color_img).save(color_dir / f"{stem}.png")

    overlay_dir = out_dir / "overlay"
    overlay_dir.mkdir(exist_ok=True)
    orig_resized = np.array(
        Image.fromarray(original_img).resize(
            (pred_mask.shape[1], pred_mask.shape[0]), Image.BILINEAR
        )
    )
    overlay = (0.5 * orig_resized + 0.5 * color_img).astype(np.uint8)
    Image.fromarray(overlay).save(overlay_dir / f"{stem}.png")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"[test] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("[test] Using CPU")

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not Path(CHECKPOINT).exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT}\n"
            "Make sure train.py finished successfully first."
        )

    model = build_model(num_classes=NUM_CLASSES).to(device)
    ckpt  = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[test] Loaded checkpoint (trained for {ckpt.get('epoch','?')} epochs, "
          f"best mIoU={ckpt.get('best_iou', 0):.4f})")

    test_ds = DesertTestDataset(TEST_DIR, img_size=IMG_SIZE)
    loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS)

    raw_image_dir = Path(TEST_DIR) / "Color_Images"
    if not raw_image_dir.exists():
        raw_image_dir = Path(TEST_DIR)

    processed = 0
    with torch.no_grad():
        for images, stems in loader:
            images = images.to(device)
            logits = model(images)
            preds  = logits.argmax(dim=1).cpu().numpy()

            for i, stem in enumerate(stems):
                pred_mask = preds[i]

                original = None
                for ext in [".png", ".jpg", ".jpeg"]:
                    img_path = raw_image_dir / f"{stem}{ext}"
                    if img_path.exists():
                        original = np.array(Image.open(img_path).convert("RGB"))
                        break
                if original is None:
                    original = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)

                save_predictions(pred_mask, original, stem, out_dir)
                processed += 1

    print(f"\n[test] Done! {processed} images processed.")
    print(f"[test] Results saved to: {out_dir}")
    print(f"  C:\\hackathon_outputs\\predictions\\color\\    ")
    print(f"  C:\\hackathon_outputs\\predictions\\overlay\\  ")
    print(f"  C:\\hackathon_outputs\\predictions\\raw\\      ")


if __name__ == "__main__":
    main()

