
import os
from pathlib import Path
from PIL import Image
from skimage.filters import threshold_multiotsu
from skimage import morphology, color
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
import matplotlib.pyplot as plt

import segmentation_models_pytorch as smp

def resize_with_padding(img, image_size, is_mask=False):
    W, H = img.size

    scale = min(image_size[0] / H, image_size[1] / W, 1.0)

    new_h = int(H * scale)
    new_w = int(W * scale)

    if scale < 1:
        if is_mask:

            img_bicubic = img.resize((new_w, new_h),resample=Image.Resampling.BICUBIC)
            img_arr = np.asarray(img_bicubic.convert("RGB"))
            gray = color.rgb2gray(img_arr)

            thresholds = threshold_multiotsu(gray, classes=3)
            regions = np.digitize(gray, bins=thresholds)
            img_resized_bool = regions >= 1
            img_resized = Image.fromarray((img_resized_bool.astype(np.uint8) * 255), mode = "L")

        else:
            resample = Image.BILINEAR
            img_resized = img.resize((new_w, new_h), resample=resample)

    else:
        img_resized = img.copy()

    pad_h = image_size[0] - new_h #How much padding is needed for height
    pad_w = image_size[1] - new_w #how much for width

    pad_top = pad_h // 2
    pad_left = pad_w // 2

    if is_mask:
        new_img = Image.new("L", (image_size[1], image_size[0]), 0)
    else:
        new_img = Image.new("RGB", (image_size[1], image_size[0]), (0, 0, 0))

    new_img.paste(img_resized, (pad_left, pad_top))

    return new_img

class SlimeMoldDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.image_size = image_size

        self.image_names = sorted([
            f.name for f in self.image_dir.iterdir()
            if f.suffix.lower() in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]
        ])

        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        name = self.image_names[idx]
        image_path = self.image_dir/name

        stem = Path(name).stem #This extracts the name of the file without the extension
        mask_name = stem + "_Network.png" #Maybe make it more general to accept any suffix
        mask_path = self.mask_dir / mask_name

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = resize_with_padding(image, self.image_size, is_mask = False)
        mask = resize_with_padding(mask, self.image_size, is_mask=True)

        image = self.to_tensor(image)
        mask = self.to_tensor(mask)

        mask = mask.float()

        return image, mask
    
def dice_coefficient(logits, true_pixels, eps=1e-6): #Avg. Dice coefficient per batch
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()

    intersection = (preds * true_pixels).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + true_pixels.sum(dim=(1, 2, 3)) 

    dice = (2.0 * intersection + eps) / (union + eps)
    return dice.mean()


def train_one_epoch(model, loader, optimizer, myLoss, device):
    model.train()
    running_loss = 0.0
    running_dice = 0.0
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = myLoss(logits, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_dice += dice_coefficient(logits, masks).item()

    return running_loss / len(loader), running_dice / len(loader)

@torch.no_grad()
def evaluate(model, loader, myLoss, device):
    model.eval()
    running_loss = 0.0
    running_dice = 0.0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        loss = myLoss(logits, masks)

        running_loss += loss.item()
        running_dice += dice_coefficient(logits, masks).item()

    return running_loss / len(loader), running_dice / len(loader)

img_size = (3200, 3200)

data_dir = Path(os.environ.get("SLIME_DATA_DIR", "Data"))
images_path = data_dir / "images"
masks_path = data_dir / "network_filled"

dataset = SlimeMoldDataset(image_dir = images_path, mask_dir = masks_path, image_size = img_size)
n_total = len(dataset)
n_train = int(0.8 * n_total)
n_val = n_total - n_train


train_dataset, val_dataset = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))

batch_size = 3

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1).to(device)

myLoss = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
num_epochs = 40

history = {
    "train_loss": [],
    "train_dice": [],
    "val_loss": [],
    "val_dice": [],
}

best_val_loss = float("inf")
results_dir = Path("outputs") / "resized"
results_dir.mkdir(parents=True, exist_ok=True)
save_model_path = results_dir / "best_resized_3200.pth"

for epoch in range(num_epochs):
    print(f"\n--- epoch {epoch+1}/{num_epochs}")
    train_loss, train_dice = train_one_epoch(model, train_loader, optimizer, myLoss, device)
    val_loss, val_dice = evaluate(model, val_loader, myLoss, device)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), save_model_path)

    history["train_loss"].append(train_loss)
    history["train_dice"].append(train_dice)
    history["val_loss"].append(val_loss)
    history["val_dice"].append(val_dice)

    print(
        f"Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f} | "
        f"Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}"
    )

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history["train_loss"],label="train loss")
plt.plot(history["val_loss"],label="val loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(f"Loss resized images model {img_size}")
plt.legend()
plt.grid(True)
plt.ylim(0, 1.7)

plt.subplot(1,2,2)
plt.plot(history["train_dice"],label="train dice")
plt.plot(history["val_dice"],label="val dice")
plt.xlabel("Epoch")
plt.ylabel("Dice")
plt.title(f"Dice Coef. resized images model {img_size}")
plt.legend()
plt.grid(True)
plt.ylim(0, 1)

plt.tight_layout()

plot_path = results_dir / "metrics_resized_3200.png"
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.close()