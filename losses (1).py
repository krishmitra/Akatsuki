import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):

    def __init__(self, num_classes: int, smooth: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        one_hot = F.one_hot(targets, self.num_classes)
        one_hot = one_hot.permute(0, 3, 1, 2).float()

        probs   = probs.view(probs.size(0), self.num_classes, -1)
        one_hot = one_hot.view(one_hot.size(0), self.num_classes, -1)

        intersection = (probs * one_hot).sum(dim=2)
        union        = probs.sum(dim=2) + one_hot.sum(dim=2)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):

    def __init__(self, num_classes: int, weight: torch.Tensor = None,
                 ce_weight: float = 0.6, dice_weight: float = 0.4):
        super().__init__()
        self.ce   = nn.CrossEntropyLoss(weight=weight)
        self.dice = DiceLoss(num_classes)
        self.ce_w   = ce_weight
        self.dice_w = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.ce_w * self.ce(logits, targets) + \
               self.dice_w * self.dice(logits, targets)


def compute_iou(preds: torch.Tensor, targets: torch.Tensor,
                num_classes: int) -> dict:

    iou_per_class = []

    preds   = preds.view(-1)
    targets = targets.view(-1)

    for cls in range(num_classes):
        pred_c   = (preds   == cls)
        target_c = (targets == cls)

        intersection = (pred_c & target_c).sum().item()
        union        = (pred_c | target_c).sum().item()

        if union == 0:
            iou_per_class.append(float("nan"))
        else:
            iou_per_class.append(intersection / union)

    valid = [v for v in iou_per_class if not (v != v)]
    mean_iou = sum(valid) / len(valid) if valid else 0.0

    return {"per_class": iou_per_class, "mean_iou": mean_iou}


