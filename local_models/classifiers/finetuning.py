import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from celeb_resnet_model import AttributeClassifier, CelebADataset, evaluate, TrainingHistory, EarlyStopping
import copy
import random
from celeb_configs import dataset_path, finetuning_data_path
from itertools import cycle

print(os.getcwd())
# ============ config  ============
TARGET_IDX = 15                     # Eyeglasses Index

GEN_LABEL_FILE = os.path.join(finetuning_data_path, 'dataset_labels_done.csv') 
OLD_MODEL_PATH = "./local_models/classifiers/checkpoints/resnet_celeb_40_single.pth"
SAVE_PATH = "./local_models/classifiers/checkpoints/resnet_finetuned_eyeglasses.pth"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
lr = 2e-5
num_epochs = 20
lambda_kd = 0.0  # consistency  0.3
mix_ratio = 0.8  # 80% original, 20% generated
num_workers = 4
patience = 10
verbose = True

#%% ============ data transform ============
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    #transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

#%% ============ Gen dataset ============
class EyeglassesGeneratedDataset(Dataset):
    def __init__(self, image_dir, label_csv, transform=None, split='train', test_ratio=0.2, seed=25):
        import pandas as pd
        # accept either a path or a DataFrame
        if isinstance(label_csv, pd.DataFrame):
            df_all = label_csv.copy()
        else:
            df_all = pd.read_csv(label_csv)

        self.image_dir = image_dir
        self.transform = transform

        if 0.0 < test_ratio < 1.0:
            test_df = df_all.sample(frac=test_ratio, random_state=seed)
            train_df = df_all.drop(test_df.index)
        else:
            train_df = df_all
            test_df = pd.DataFrame(columns=df_all.columns)

        self.df = train_df.reset_index(drop=True) if split == 'train' else test_df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        from PIL import Image
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row["image_path"].split('/')[-1])
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = torch.zeros(40, dtype=torch.float)
        label[TARGET_IDX] = torch.tensor(row["label"], dtype=torch.float)
        # Since labels are -1/1, convert into 0/1 for BCEWithLogitsLoss
        label = (label + 1) // 2
        return img, label

# create a generated test split (20%) accessible as gen_test_dataset
gen_test_dataset = EyeglassesGeneratedDataset(finetuning_data_path, GEN_LABEL_FILE, transform, split='test', test_ratio=0.2, seed=42)

#%%  ============ load ============
real_dataset = CelebADataset(root=dataset_path, split='train', transform=transform)
gen_dataset = EyeglassesGeneratedDataset(finetuning_data_path, GEN_LABEL_FILE, transform)
gen_test_dataset = EyeglassesGeneratedDataset(finetuning_data_path, GEN_LABEL_FILE, transform, split='test', test_ratio=0.2, seed=42)

val_dataset = CelebADataset(root=dataset_path,
                            transform=None,  # use default
                            split='val'
                            )

real_loader = DataLoader(real_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
gen_loader = DataLoader(gen_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

test_gen_loader = DataLoader(gen_test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


#%% ============ model ============
model = AttributeClassifier(num_classes=40, pretrained=False).to(device)
old_model = copy.deepcopy(model).eval()  
#%% load pre-trained weights
checkpoint = torch.load(OLD_MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint)
old_model.load_state_dict(checkpoint)

# Enable multi-GPU training if available
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)
    old_model = nn.DataParallel(old_model)

# Ensure the model is on CUDA before creating optimizer


# Freeze backbone
for name, param in model.named_parameters():
    if "backbone.fc" not in name:
        param.requires_grad = False

# Only optimize fc layer
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
criterion = nn.BCEWithLogitsLoss()

early_stopping = EarlyStopping(patience=patience, verbose=verbose)

gen_iter = cycle(gen_loader)  # Cycle through generated samples to prevent length issues with zip

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for i, real_batch in enumerate(real_loader):
        # Get one batch from gen_iter at each step to construct mixed batch
        gen_batch = next(gen_iter)

        # Sample from real/gen in proportion to construct mixed batch
        real_imgs, real_labels = real_batch
        gen_imgs, gen_labels = gen_batch

        # Ensure consistent size (can crop or repeat generated samples)
        mix_size = int(batch_size * (1.0 - mix_ratio))   # Number of generated samples
        mix_size = max(1, mix_size)
        # Randomly sample mix_size generated samples
        idxs = torch.randperm(gen_imgs.size(0))[:mix_size]
        gen_part_imgs = gen_imgs[idxs]
        gen_part_labels = gen_labels[idxs]

        # Rest from real
        real_part_size = batch_size - mix_size
        ridx = torch.randperm(real_imgs.size(0))[:real_part_size]
        real_part_imgs = real_imgs[ridx]
        real_part_labels = real_labels[ridx]

        # Merge
        # Concatenate and shuffle together
        images = torch.cat([real_part_imgs, gen_part_imgs], dim=0)
        labels = torch.cat([real_part_labels, gen_part_labels], dim=0)
        
        # Generate random permutation
        perm = torch.randperm(images.size(0))
        images = images[perm].to(device)
        labels = labels[perm].to(device)

        logits = model(images)
        logits_old = old_model(images).detach()

        # Loss for eyeglasses only
        loss_eyeg = criterion(logits[:, TARGET_IDX], labels[:, TARGET_IDX])

        # KD on other dimensions (if lambda_kd > 0)
        other_idx = [i for i in range(40) if i != TARGET_IDX]
        loss_kd = torch.nn.functional.mse_loss(logits[:, other_idx], logits_old[:, other_idx])
        loss = loss_eyeg + lambda_kd * loss_kd

        optimizer.zero_grad()
        loss.backward()

        # ---------- Corrected mask (mask by row, preserve target row) ----------
        with torch.no_grad():
            w_grad = model.module.backbone.fc.weight.grad  # shape (40, in_features)
            if w_grad is not None:
                mask = torch.zeros_like(w_grad)
                mask[TARGET_IDX, :] = 1.0
                model.module.backbone.fc.weight.grad *= mask

            b_grad = model.module.backbone.fc.bias.grad  # shape (40,)
            if b_grad is not None:
                b_mask = torch.zeros_like(b_grad)
                b_mask[TARGET_IDX] = 1.0
                model.module.backbone.fc.bias.grad *= b_mask

        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(gen_loader)
    # ============ validation ============
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    val_loss_target, val_acc_target = evaluate(model, val_loader, criterion, device, target_logit=TARGET_IDX)
    loss_t, acc_t = evaluate(model, test_gen_loader, criterion, device, target_logit=TARGET_IDX)
    if verbose:
        print(f"[Epoch {epoch+1}/{num_epochs}] Train Loss: {avg_loss:.4f} Val Loss:{val_loss:.4f} Val Acc: {val_acc:.4f} Target Acc: {val_acc_target:.4f}")
        print(f"Test Loss: {loss_t:.4f} Acc: {acc_t:.4f}")
    early_stopping(val_loss, model, SAVE_PATH)
    if early_stopping.early_stop:
        print("Early stopping triggered")
        break

#torch.save(model.state_dict(), SAVE_PATH)
print(f" Fine-tuned model saved to {SAVE_PATH}")


# ============ test on generated test set ============
# Evaluate old vs finetuned on CelebA test set and generated test set
test_real_dataset = CelebADataset(root=dataset_path, split='test', transform=None)
test_real_loader = DataLoader(test_real_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

def _eval_and_print(name, loader, loader_name):
    model_to_eval = old_model if name == "Old" else model
    model_to_eval.eval()
    model_to_eval.to(device)
    with torch.no_grad():
        loss, acc = evaluate(model_to_eval, loader, criterion, device)
        loss_t, acc_t = evaluate(model_to_eval, loader, criterion, device, target_logit=TARGET_IDX)
    print(f"[{name} on {loader_name}] Loss: {loss:.4f} Acc: {acc:.4f} Target Acc(idx={TARGET_IDX}): {acc_t:.4f}")

print("\n=== Comparison on real CelebA test set ===")
_eval_and_print("Old", test_real_loader, "CelebA-test")
_eval_and_print("Finetuned", test_real_loader, "CelebA-test")

print("\n=== Comparison on generated test set ===")
_eval_and_print("Old", test_gen_loader, "Generated-test")
_eval_and_print("Finetuned", test_gen_loader, "Generated-test")