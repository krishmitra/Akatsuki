import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

LABEL_MAP = {
    0:     0,
    100:   1,
    200:   2,
    300:   3,
    500:   4,
    550:   5,
    700:   6,
    800:   7,
    7100:  8,
    10000: 9,
}
NUM_CLASSES = 10

CLASS_COLORS = [
    (0,   0,   0),
    (34,  139, 34),
    (0,   255, 0),
    (210, 180, 140),
    (139, 90,  43),
    (128, 128, 128),
    (101, 67,  33),
    (169, 169, 169),
    (135, 206, 235),
    (255, 255, 255),
]


def remap_mask(mask_np: np.ndarray) -> np.ndarray:
    out = np.zeros(mask_np.shape, dtype=np.int16)
    for raw_val, idx in LABEL_MAP.items():
        out[mask_np == raw_val] = idx
    return out


def get_train_augmentation(img_size: int = 512):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.HueSaturationValue(p=0.2),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                           rotate_limit=15, p=0.4),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_augmentation(img_size: int = 512):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


class DesertSegDataset(Dataset):

    def __init__(self, split_dir: str, transform=None):
        self.image_dir = Path(split_dir) / "Color_Images"
        self.mask_dir  = Path(split_dir) / "Segmentation"
        self.transform = transform

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image dir not found: {self.image_dir}")
        if not self.mask_dir.exists():
            raise FileNotFoundError(f"Mask dir not found: {self.mask_dir}")

        valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        images = {p.stem: p for p in self.image_dir.iterdir()
                  if p.suffix.lower() in valid_exts}
        masks  = {p.stem: p for p in self.mask_dir.iterdir()
                  if p.suffix.lower() in valid_exts}

        common = sorted(set(images.keys()) & set(masks.keys()))
        if len(common) == 0:
            raise RuntimeError(
                f"No matching image/mask pairs found in {split_dir}.\n"
                f"  Images: {len(images)}, Masks: {len(masks)}"
            )

        self.pairs = [(images[s], masks[s]) for s in common]
        print(f"[dataset] Loaded {len(self.pairs)} pairs from: {split_dir}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        image = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        mask  = np.array(Image.open(mask_path).convert("I"),  dtype=np.int32)

        mask = remap_mask(mask.astype(np.int64)).astype(np.int64)

        if self.transform:
            aug = self.transform(image=image, mask=mask)
            image = aug["image"]
            mask  = aug["mask"].long()

        return image, mask


class DesertTestDataset(Dataset):

    def __init__(self, test_dir: str, img_size: int = 512):
        self.image_dir = Path(test_dir) / "Color_Images"
        if not self.image_dir.exists():
            self.image_dir = Path(test_dir)

        valid_exts = {".png", ".jpg", ".jpeg", ".bmp"}
        self.paths = sorted(
            p for p in self.image_dir.iterdir()
            if p.suffix.lower() in valid_exts
        )
        if len(self.paths) == 0:
            raise RuntimeError(f"No images found in {self.image_dir}")

        self.transform = get_val_augmentation(img_size)
        print(f"[dataset] Test images found: {len(self.paths)}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        image = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        aug   = self.transform(image=image)
        return aug["image"], img_path.stem


