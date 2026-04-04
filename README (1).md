# Desert Semantic Segmentation — U-Net Solution

End-to-end U-Net solution for multi-class semantic segmentation of off-road desert environments.

---

## Project Structure

```
desert_segmentation/
│
├── data/                          ← Your dataset goes here
│   ├── train/
│   │   ├── images/               ← Training RGB images (.jpg / .png)
│   │   └── masks/                ← Corresponding segmentation masks (.png)
│   └── test/
│       └── images/               ← Test images only (no masks!)
│
├── outputs/                       ← Auto-created during training
│   ├── checkpoints/
│   │   ├── best.pth              ← Best model by val mIoU
│   │   └── last.pth              ← Latest checkpoint
│   ├── predictions/
│   │   ├── raw_masks/            ← Class-index PNGs
│   │   ├── colored/              ← Color-coded visualizations
│   │   └── overlay/              ← Image + mask blends
│   ├── label_map.json            ← Raw pixel value → class index mapping
│   └── train_history.json        ← Per-epoch metrics log
│
├── dataset.py                     ← Dataset, transforms, label remapping
├── model.py                       ← U-Net with ResNet-34 encoder
├── losses.py                      ← CombinedLoss (CE + Dice), IoU metric
├── train.py                       ← Training + validation pipeline
├── test.py                        ← Inference on test images
├── utils.py                       ← Dataset splitting, class weights, eval
├── requirements.txt
└── README.md
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Data Preparation

Place your data under `data/`:

```
data/train/images/   ← all training images
data/train/masks/    ← all training masks (same filename stems)
data/test/images/    ← test images (NO masks — never used during training)
```

Masks can contain **any pixel values** (sequential or not). The code automatically
scans all masks, discovers unique values, and builds a remapping to sequential
class indices (0, 1, 2, …).

---

## Training

```bash
python train.py \
  --data_root   data \
  --output_dir  outputs \
  --img_size    512 \
  --epochs      50 \
  --batch_size  8 \
  --lr          1e-3 \
  --val_split   0.15
```

Key flags:
| Flag | Default | Description |
|------|---------|-------------|
| `--epochs` | 50 | Training epochs |
| `--batch_size` | 8 | Reduce if OOM |
| `--img_size` | 512 | Resize all images to this |
| `--val_split` | 0.15 | Fraction held out for validation |
| `--amp` | True | Automatic mixed precision (faster on GPU) |
| `--resume` | "" | Path to checkpoint to resume from |

---

## Inference (Test Set)

```bash
python test.py \
  --checkpoint  outputs/checkpoints/best.pth \
  --test_dir    data/test/images \
  --output_dir  outputs/predictions
```

Saves three sets of outputs:
- `raw_masks/`  — Single-channel PNGs with class indices (submit these)
- `colored/`    — Human-readable color visualizations
- `overlay/`    — Original image blended with colored mask

---

## Architecture

```
Input [B, 3, H, W]
  │
  ├─ ResNet-34 Encoder (pretrained ImageNet)
  │     enc0: stem   → [B,  64, H/2,  W/2]
  │     enc1: layer1 → [B,  64, H/4,  W/4]
  │     enc2: layer2 → [B, 128, H/8,  W/8]
  │     enc3: layer3 → [B, 256, H/16, W/16]
  │     enc4: layer4 → [B, 512, H/32, W/32]  ← bottleneck
  │
  └─ U-Net Decoder (skip connections)
        dec4: 512→256, skip=enc3 → [B, 256, H/16, W/16]
        dec3: 256→128, skip=enc2 → [B, 128, H/8,  W/8]
        dec2: 128→ 64, skip=enc1 → [B,  64, H/4,  W/4]
        dec1:  64→ 32, skip=enc0 → [B,  32, H/2,  W/2]
        final: upsample×2 + conv1×1 → [B, num_classes, H, W]
```

**~21M parameters total** — fast to train, strong baseline.

---

## Loss Function

`CombinedLoss = 0.6 × CrossEntropyLoss + 0.4 × SoftDiceLoss`

- Cross-entropy handles per-pixel classification.
- Dice loss encourages overlap, especially for minority classes.

---

## Metrics

- **mIoU** (mean Intersection over Union) — primary metric
- **Pixel Accuracy** — secondary

---

## Tips for Better Results

1. **Class imbalance** — use `compute_class_weights()` from `utils.py` and pass to `CombinedLoss(class_weights=...)`.
2. **More epochs** — try 80–100 for best results.
3. **Larger encoder** — swap ResNet-34 → ResNet-50 in `model.py` for ~+1–2% mIoU.
4. **TTA** — test-time augmentation (horizontal flip) can boost mIoU by ~0.5–1%.
5. **img_size=640** — larger resolution for fine-grained desert textures.
