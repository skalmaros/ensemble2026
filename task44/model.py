import os
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Urządzenie: {device}")

# ============================================================================
# KONFIGURACJA
# ============================================================================

TARGET_WIDTH = 1024
TARGET_HEIGHT = 768
BATCH_SIZE = 8
EPOCHS = 75
LR = 1e-4
VAL_SPLIT = 0.1
IMAGES_DIR = "data/train_wyprostowane"
MASKS_DIR = "data/train_maski"
TIMEOUT_SECONDS = 14 * 60
CHECKPOINT_DIR = "checkpoints"
RESUME_FROM = "checkpoints/ostatni.pth"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ============================================================================
# AUGMENTACJA — silniejsza
# ============================================================================

train_transform = A.Compose([
    A.Resize(TARGET_HEIGHT, TARGET_WIDTH),
    A.Affine(translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
             scale=(0.85, 1.15), rotate=(-5, 5),
             mode=cv2.BORDER_CONSTANT, p=0.6),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.3),
    A.GaussianBlur(blur_limit=(3, 7), p=0.3),
    A.GaussNoise(std_range=(0.02, 0.1), p=0.3),
    A.CoarseDropout(num_holes_range=(1, 3), hole_height_range=(20, 60),
                    hole_width_range=(20, 60), fill=0, p=0.2),
    A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(TARGET_HEIGHT, TARGET_WIDTH),
    A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
    ToTensorV2(),
])


# ============================================================================
# DATASET
# ============================================================================

class ECGDataset(Dataset):
    def __init__(self, images_dir, masks_dir, file_list=None, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        if file_list is not None:
            self.images = file_list
        else:
            self.images = sorted([
                f for f in os.listdir(images_dir)
                if f.endswith('.png') or f.endswith('.jpg')
            ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        mask_name = os.path.splitext(img_name)[0] + '.png'

        image = cv2.imread(os.path.join(self.images_dir, img_name))
        mask = cv2.imread(os.path.join(self.masks_dir, mask_name), cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise ValueError(f"Brak zdjęcia: {img_name}")
        if mask is None:
            raise ValueError(f"Brak maski: {mask_name}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = (mask > 127).astype(np.uint8)

        if self.transform:
            aug = self.transform(image=image, mask=mask)
            return aug['image'].float(), aug['mask'].unsqueeze(0).float()
        else:
            image = cv2.resize(image, (TARGET_WIDTH, TARGET_HEIGHT))
            mask = cv2.resize(mask, (TARGET_WIDTH, TARGET_HEIGHT),
                              interpolation=cv2.INTER_NEAREST)
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask).unsqueeze(0).float()
            return image, mask


# ============================================================================
# PODZIAŁ TRAIN/VAL — TEN SAM SEED
# ============================================================================

all_files = sorted([
    f for f in os.listdir(IMAGES_DIR)
    if f.endswith('.png') or f.endswith('.jpg')
])

n_val = max(int(len(all_files) * VAL_SPLIT), 1)
n_train = len(all_files) - n_val

indices = list(range(len(all_files)))
np.random.seed(42)
np.random.shuffle(indices)

train_files = [all_files[i] for i in indices[:n_train]]
val_files = [all_files[i] for i in indices[n_train:]]

train_dataset = ECGDataset(IMAGES_DIR, MASKS_DIR, file_list=train_files, transform=train_transform)
val_dataset = ECGDataset(IMAGES_DIR, MASKS_DIR, file_list=val_files, transform=val_transform)

print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=8, pin_memory=True, persistent_workers=True
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=4, pin_memory=True, persistent_workers=True
)


# ============================================================================
# MODEL
# ============================================================================

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=1,
).to(device)


# ============================================================================
# LOSS: BCE + Dice + Focal
# ============================================================================

class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def focal_loss(self, pred, target, alpha=0.25, gamma=2.0):
        bce = nn.functional.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce)
        return (alpha * (1 - pt) ** gamma * bce).mean()

    def dice_loss(self, pred, target):
        pred_sig = torch.sigmoid(pred)
        smooth = 1.0
        intersection = (pred_sig * target).sum(dim=(2, 3))
        union = pred_sig.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return 1.0 - dice.mean()

    def forward(self, pred, target):
        return 0.3 * self.bce(pred, target) + 0.4 * self.dice_loss(pred, target) + 0.3 * self.focal_loss(pred, target)


criterion = CombinedLoss()

optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)


# ============================================================================
# WZNOWIENIE Z CHECKPOINTU
# ============================================================================

start_epoch = 0

if RESUME_FROM and os.path.exists(RESUME_FROM):
    print(f"Wznawiam z: {RESUME_FROM}")
    checkpoint = torch.load(RESUME_FROM, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    # Nowy optimizer i scheduler — nie ładuj starych (nowy LR)
    start_epoch = checkpoint['epoch']
    print(f"Wznowiono od epoki {start_epoch + 1}, nowy LR={LR}")
elif RESUME_FROM:
    print(f"UWAGA: {RESUME_FROM} nie istnieje, zaczynam od zera")


# ============================================================================
# METRYKI
# ============================================================================

def compute_metrics(pred, target, threshold=0.5):
    pred_bin = (torch.sigmoid(pred) > threshold).float()
    smooth = 1.0
    intersection = (pred_bin * target).sum(dim=(2, 3))
    union_iou = pred_bin.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - intersection
    union_dice = pred_bin.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    iou = ((intersection + smooth) / (union_iou + smooth)).mean().item()
    dice = ((2.0 * intersection + smooth) / (union_dice + smooth)).mean().item()
    return iou, dice


# ============================================================================
# ZAPIS CHECKPOINTU
# ============================================================================

def save_checkpoint(epoch, model, optimizer, scheduler, val_dice, is_best=False):
    state = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'val_dice': val_dice,
    }

    torch.save(state, os.path.join(CHECKPOINT_DIR, "ostatni.pth"))

    if is_best:
        torch.save(state, os.path.join(CHECKPOINT_DIR, "best.pth"))
        torch.save(model.state_dict(), "unet_best.pth")
        print(f"  >>> Nowy najlepszy! Val Dice: {val_dice:.4f}", flush=True)


# ============================================================================
# TRENING Z TIMEOUTEM
# ============================================================================

def train_model(model, train_loader, val_loader, epochs, start_epoch=0):
    best_val_dice = 0.0
    t_start = time.time()

    for epoch in range(start_epoch, epochs):
        elapsed = time.time() - t_start
        t_epoch = time.time()

        # ---- TRAIN ----
        model.train()
        train_loss = 0.0
        train_dice = 0.0

        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            preds = model(images)
            loss = criterion(preds, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, dice = compute_metrics(preds, masks)
            train_dice += dice

        scheduler.step()

        n_tb = len(train_loader)
        train_loss /= n_tb
        train_dice /= n_tb

        # ---- VALIDATION ----
        model.eval()
        val_loss = 0.0
        val_dice = 0.0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                preds = model(images)
                loss = criterion(preds, masks)
                val_loss += loss.item()
                _, dice = compute_metrics(preds, masks)
                val_dice += dice

        n_vb = len(val_loader)
        val_loss /= n_vb
        val_dice /= n_vb

        epoch_time = time.time() - t_epoch
        total_elapsed = time.time() - t_start
        remaining = TIMEOUT_SECONDS - total_elapsed

        lr_now = optimizer.param_groups[0]['lr']

        print(
            f"Epoka {epoch+1:2d}/{epochs} | "
            f"{epoch_time:.0f}s | "
            f"LR: {lr_now:.6f} | "
            f"Train L:{train_loss:.4f} D:{train_dice:.4f} | "
            f"Val L:{val_loss:.4f} D:{val_dice:.4f} | "
            f"Zostało: {remaining:.0f}s",
            flush=True
        )

        is_best = val_dice > best_val_dice
        if is_best:
            best_val_dice = val_dice
        save_checkpoint(epoch + 1, model, optimizer, scheduler, val_dice, is_best)

    print(f"\nNajlepszy Val Dice: {best_val_dice:.4f}")


# ============================================================================
# START
# ============================================================================

print(f"\nKonfiguracja:")
print(f"  Rozmiar: {TARGET_WIDTH}x{TARGET_HEIGHT}")
print(f"  Batch: {BATCH_SIZE}")
print(f"  Epoki: {start_epoch+1}..{EPOCHS}")
print(f"  LR: {LR} -> 1e-6 (cosine)")
print(f"  Loss: BCE(0.3) + Dice(0.4) + Focal(0.3)")
print(f"  Timeout: {TIMEOUT_SECONDS}s")
print()

train_model(model, train_loader, val_loader, EPOCHS, start_epoch)