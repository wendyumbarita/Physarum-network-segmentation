import os
from pathlib import Path
from PIL import Image

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import random

#PATCH MODEL USING BCE LOSS AND DATA CAN BE MODIFIED TO AGUMENTED WHEN CALLING THE TRAINING DATASET

def get_patch_starts(size, patch_size, stride): #Where each patch starts 
    if size <= patch_size:
        return [0]

    starts = list(range(0, size - patch_size + 1, stride))

    last_start = size - patch_size
    if starts[-1] != last_start: #This is to add the last patch of the image if it is not covered in "starts" because the last piece of image is smaller than the patch size
        starts.append(last_start)

    return starts


def pad_if_smaller(img, patch_size, is_mask=False):

    W, H = img.size

    new_W = max(W, patch_size)
    new_H = max(H, patch_size)

    if new_W == W and new_H == H: #if not padding needed, then returns the image as it is
        return img

    if is_mask:
        canvas = Image.new("L", (new_W, new_H), 0)
    else:
        canvas = Image.new("RGB", (new_W, new_H), (0, 0, 0))

    canvas.paste(img, (0, 0))
    return canvas


class SlimeMoldPatchDataset(Dataset):
    def __init__(
        self,
        image_dir,
        mask_dir,
        image_names=None,
        patch_size=1280,
        stride=1024,
        augment=False,
        filter_empty=False,
        min_mask_fraction=0.0001,
        empty_keep_probability=0.25,
        seed=42
    ):
        
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)

        self.patch_size = patch_size
        self.stride = stride
        self.augment = augment
        self.filter_empty = filter_empty
        self.min_mask_fraction = min_mask_fraction
        self.empty_keep_probability = empty_keep_probability

        self.rng = random.Random(seed)

        valid_extensions = [".png", ".jpg", ".jpeg", ".tif", ".tiff"]

        if image_names is None:
            self.image_names = sorted([
                f.name
                for f in self.image_dir.iterdir()
                if f.suffix.lower() in valid_extensions
            ])
        else:
            self.image_names = sorted(image_names)

        self.samples = [] #saves the information of the patches, not the patches but the info. Which image and mask they belong to and the coordinates where each patch starts in the image
        self.num_informative_patches = 0
        self.num_empty_patches_kept = 0 #We will keep only a percentage of the images that do not have too much pixels on it (min_mask_fraction) to avoid learning too much background
        self.num_empty_patches_removed = 0

        for name in self.image_names:
            image_path = self.image_dir / name
            stem = Path(name).stem
            mask_path = self.mask_dir / f"{stem}_Network.png"

            if not image_path.exists():
                print(f"Image not found: {image_path}")
                continue

            if not mask_path.exists():
                print(f"Mask not found for {name}: {mask_path}")
                continue

            with Image.open(image_path) as image:
                W, H = image.size

            with Image.open(mask_path) as mask:
                mask = mask.convert("L")

                if mask.size != (W, H):
                    raise ValueError(
                        "Image and mask sizes do not match:\n"
                        f"{image_path}: {(W, H)}\n"
                        f"{mask_path}: {mask.size}"
                    )

                mask_padded = pad_if_smaller(mask,patch_size,is_mask=True)

                W_pad, H_pad = mask_padded.size
                x_starts = get_patch_starts(W_pad,patch_size,stride)
                y_starts = get_patch_starts(H_pad,patch_size,stride)

                for y in y_starts:
                    for x in x_starts:
                        keep_patch = True

                        if self.filter_empty:

                            box = (x,y,x + patch_size,y + patch_size) #patch_box 
                            mask_patch = mask_padded.crop(box)
                            mask_arr = np.asarray(mask_patch,dtype=np.uint8)
                            mask_fraction = np.mean(mask_arr > 127)

                            if mask_fraction >= self.min_mask_fraction:
                                self.num_informative_patches += 1
                            else: #Here is where patches with no so many pixels are chosen to be kept or removed according to the probability given 
                                keep_patch = (self.rng.random()< self.empty_keep_probability)
                                if keep_patch:
                                    self.num_empty_patches_kept += 1
                                else:
                                    self.num_empty_patches_removed += 1

                        else:
                            self.num_informative_patches += 1

                        if keep_patch:
                            self.samples.append((image_path,mask_path,x,y))

        if self.filter_empty:

            print(f"\nDataset: {self.image_dir}")
            print("Informative patches: "f"{self.num_informative_patches}")
            print("Empty patches kept: "f"{self.num_empty_patches_kept}")
            print("Empty patches removed: " f"{self.num_empty_patches_removed}")
            print(f"Final number of patches: {len(self.samples)}")

    def __len__(self):
        return len(self.samples) #Total of patches in the whole dataset

    def __getitem__(self, idx):
        image_path, mask_path, x, y = self.samples[idx]

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = pad_if_smaller(image,self.patch_size,is_mask=False)
        mask = pad_if_smaller(mask,self.patch_size,is_mask=True)

        box = (x,y,x + self.patch_size,y + self.patch_size)

        image_patch = image.crop(box)
        mask_patch = mask.crop(box)

        image_arr = (np.asarray(image_patch).astype(np.float32)/ 255.0)
        mask_arr = np.asarray(mask_patch).astype(np.float32)
        mask_arr = (mask_arr > 127).astype(np.float32)

        # Random rotation:
        # k = 0, 1, 2 or 3 corresponds to
        # 0°, 90°, 180° or 270°.

        if self.augment: #This will give us a different rotation every time the patch is called. Let's say in the epoch 1 the rotation was 90 deg, in the next epoch the rotation can be different (180 deg) 
            k = random.randint(0, 3)
            image_arr = np.rot90(image_arr,k=k,axes=(0, 1)).copy()
            mask_arr = np.rot90(mask_arr,k=k,axes=(0, 1)).copy() 

        image_tensor = torch.from_numpy(image_arr).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask_arr).unsqueeze(0)

        return image_tensor, mask_tensor

def get_valid_image_names(image_dir, mask_dir):
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)

    valid_extensions = [".png", ".jpg", ".jpeg", ".tif", ".tiff"]
    valid_names = []

    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in valid_extensions:
            continue

        mask_path = mask_dir / f"{image_path.stem}_Network.png"

        if mask_path.exists():
            valid_names.append(image_path.name)
        else:
            print(
                f"Skipping {image_path.name}: "
                f"mask not found"
            )

    return valid_names

def split_image_names(image_names, train_fraction=0.8, seed=42):
    
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(image_names), generator=generator).tolist()

    shuffled_names = [
        image_names[i]
        for i in permutation
    ]

    n_train = int(train_fraction * len(shuffled_names))

    train_names = shuffled_names[:n_train]
    val_names = shuffled_names[n_train:]

    return train_names, val_names

def dice_coefficient(logits, true_pixels, eps=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()

    intersection = (preds * true_pixels).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + true_pixels.sum(dim=(1, 2, 3))

    dice = (2.0 * intersection + eps) / (union + eps) #eps is to avoid dividing by zero when the mask is empty
    return dice.mean()


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()

    running_loss = 0.0
    running_dice = 0.0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = loss_fn(logits, masks)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_dice += dice_coefficient(logits, masks).item()

    return running_loss / len(loader), running_dice / len(loader)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()

    running_loss = 0.0
    running_dice = 0.0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        loss = loss_fn(logits, masks)

        running_loss += loss.item()
        running_dice += dice_coefficient(logits, masks).item()

    return running_loss / len(loader), running_dice / len(loader)




patch_size = 1280
stride = 1024
batch_size = 4
num_epochs = 40
learning_rate = 1e-4

# Use SLIME_BASE_DIR to point to a directory containing the Data/ folder.
base_dir = Path(os.environ.get("SLIME_BASE_DIR", "."))

blue_image_dir = base_dir / "Data"/ "blue_BG" / "images"
blue_mask_dir = base_dir / "Data" / "blue_BG" / "network"

yellow_image_dir = base_dir / "Data"/ "yellow_BG" / "images"
yellow_mask_dir = base_dir / "Data" / "yellow_BG" / "network"

temp_image_dir = base_dir / "Data"/ "nice_temp" / "images"
temp_mask_dir = base_dir / "Data" / "nice_temp" / "network"


#IMPORTAAAAANT: If using AUGMENTED, then need to type "True" in train_dataset. Not for the validation 

yellow_names = get_valid_image_names(yellow_image_dir,yellow_mask_dir)
yellow_train_names, yellow_val_names = split_image_names(yellow_names, train_fraction=0.8,seed=42)
yellow_train_dataset = SlimeMoldPatchDataset(image_dir=yellow_image_dir,mask_dir=yellow_mask_dir,
    image_names=yellow_train_names,patch_size=patch_size,stride=stride,augment=False,
    filter_empty=True,min_mask_fraction=0.0001,empty_keep_probability=0.25,seed=42)
yellow_val_dataset = SlimeMoldPatchDataset(image_dir=yellow_image_dir,mask_dir=yellow_mask_dir,
    image_names=yellow_val_names,patch_size=patch_size,stride=stride,augment=False,filter_empty=False)


blue_names = get_valid_image_names(blue_image_dir,blue_mask_dir)
blue_train_names, blue_val_names = split_image_names(blue_names, train_fraction=0.8,seed=43)
blue_train_dataset = SlimeMoldPatchDataset(image_dir=blue_image_dir,mask_dir=blue_mask_dir,
    image_names=blue_train_names,patch_size=patch_size,stride=stride,augment=False,
    filter_empty=True,min_mask_fraction=0.0001,empty_keep_probability=0.25,seed=43)
blue_val_dataset = SlimeMoldPatchDataset(image_dir=blue_image_dir,mask_dir=blue_mask_dir,
    image_names=blue_val_names,patch_size=patch_size,stride=stride,augment=False,filter_empty=False)


temp_names = get_valid_image_names(temp_image_dir,temp_mask_dir)
temp_train_names, temp_val_names = split_image_names(temp_names, train_fraction=0.8,seed=44)
temp_train_dataset = SlimeMoldPatchDataset(image_dir=temp_image_dir,mask_dir=temp_mask_dir,
    image_names=temp_train_names,patch_size=patch_size,stride=stride,augment=False,
    filter_empty=True,min_mask_fraction=0.0001,empty_keep_probability=0.25,seed=44)
temp_val_dataset = SlimeMoldPatchDataset(image_dir=temp_image_dir,mask_dir=temp_mask_dir,
    image_names=temp_val_names,patch_size=patch_size,stride=stride,augment=False,filter_empty=False)



train_dataset = ConcatDataset([yellow_train_dataset,blue_train_dataset,temp_train_dataset])
val_dataset = ConcatDataset([yellow_val_dataset,blue_val_dataset,temp_val_dataset])


train_loader = DataLoader(train_dataset,
    batch_size=batch_size,shuffle=True,
    num_workers=0,pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(val_dataset,
    batch_size=batch_size,shuffle=False,
    num_workers=0,pin_memory=torch.cuda.is_available()
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = smp.Unet(encoder_name="resnet34",encoder_weights="imagenet",in_channels=3,classes=1).to(device)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

results_dir = base_dir / "Models"
results_dir.mkdir(parents=True, exist_ok=True)

history = {
    "train_loss": [],
    "train_dice": [],
    "val_loss": [],
    "val_dice": [],
}

best_val_loss = float("inf")
best_val_dice = -1

for epoch in range(num_epochs):
    print(f"\n--- Epoch {epoch + 1}/{num_epochs}")

    train_loss, train_dice = train_one_epoch(
        model, train_loader, optimizer, loss_fn, device
    )

    val_loss, val_dice = evaluate(
        model, val_loader, loss_fn, device
    )

    history["train_loss"].append(train_loss)
    history["train_dice"].append(train_dice)
    history["val_loss"].append(val_loss)
    history["val_dice"].append(val_dice)

    print(
        f"Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f} | "
        f"Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), results_dir / "loss_patches_V4.pth")
    
    if val_dice > best_val_dice:
        best_val_dice = val_dice
        torch.save(model.state_dict(), results_dir / "dice_patches_V4.pth")



plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history["train_loss"], label="train loss")
plt.plot(history["val_loss"], label="val loss")
plt.legend()
plt.grid(True)
plt.ylim(0, 1)
plt.title("Loss (BCE) patches")
plt.xlabel("Epoch")

plt.subplot(1, 2, 2)
plt.plot(history["train_dice"], label="train dice")
plt.plot(history["val_dice"], label="val dice")
plt.legend()
plt.grid(True)
plt.ylim(0, 1)
plt.title("Dice Coef. patches")
plt.xlabel("Epoch")

plt.tight_layout()

plot_path = results_dir / "metrics_patches_V4.png"
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.close()
