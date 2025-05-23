from dataset import PHImageRGBDataset
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import torch.optim as optim
from model import PHRegressionModel  # 這是我們自定義的模型
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.model_selection import train_test_split

# 設定資料路徑

csv_path = 'all_labels.csv'
image_dir = 'all_data/'

# Load CSV and remove invalid test samples
df = pd.read_csv(csv_path)
df = df[(df['ph_value'] != 'testdrop') & (df['ph_value'].astype(float) > 0)]
# Write filtered CSV to temporary file
filtered_csv = 'filtered_labels.csv'
df.to_csv(filtered_csv, index=False)
# Use filtered CSV path
csv_path = filtered_csv

os.makedirs("models", exist_ok=True)

# Training-time augmentations and preprocessing
train_transform = transforms.Compose([
    transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])
# Validation/preprocessing transform (no augmentation)
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])


# Split into train/val/test with 80/10/10 ratios
df_train_val, df_test = train_test_split(df, test_size=0.10, random_state=42, shuffle=True)
df_train, df_val     = train_test_split(df_train_val, test_size=0.1111, random_state=42, shuffle=True)
# Save split CSVs (optional)
df_train.to_csv('train_labels.csv', index=False)
df_val.to_csv('val_labels.csv', index=False)
df_test.to_csv('test_labels.csv', index=False)

# Create datasets
train_dataset = PHImageRGBDataset(csv_file='train_labels.csv', image_dir=image_dir, transform=train_transform)
val_dataset   = PHImageRGBDataset(csv_file='val_labels.csv',   image_dir=image_dir, transform=val_transform)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=64, shuffle=False)

# 接下來這裡可以加上模型定義、損失函數與訓練迴圈
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# 跨訓練保留最佳模型機制
best_model_path = "models/best.pth"
best_loss = float('inf')
model = PHRegressionModel().to(device)
if os.path.exists(best_model_path):
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    print("✅ 已載入前次最佳模型")
    if os.path.exists("models/best_loss.txt"):
        with open("models/best_loss.txt", "r") as f:
            best_loss = float(f.read().strip())

# Use Huber loss per-sample for custom weighting
criterion = nn.SmoothL1Loss(reduction='none')
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=8)
early_stop_patience = 15
no_improve_epochs = 0

# Emphasize samples with true pH near 6
focus_weight = 2.0
focus_low = 5.5
focus_high = 6.5

num_epochs = 50
loss_history = []
val_loss_history = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, rgb_values, ph_values in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images = images.to(device)
        rgb_values = rgb_values.to(device)
        ph_values = ph_values.to(device).unsqueeze(1)
        ph_values = torch.clamp(ph_values, min=0.0, max=14.0)

        optimizer.zero_grad()
        outputs = model(images, rgb_values)
        # Compute per-sample Huber loss
        sample_losses = criterion(outputs, ph_values).squeeze(1)  # shape [batch]
        # Assign higher weight to samples with pH in [5.5, 6.5]
        weights = torch.where(
            (ph_values.squeeze(1) >= focus_low) & (ph_values.squeeze(1) <= focus_high),
            focus_weight,
            1.0
        )
        weighted_losses = sample_losses * weights.to(sample_losses.device)
        loss = weighted_losses.mean()
        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    loss_history.append(avg_loss)

    # ========== 驗證階段 ==========
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, rgb_values, ph_values in val_loader:
            images = images.to(device)
            rgb_values = rgb_values.to(device)
            ph_values = ph_values.to(device).unsqueeze(1)
            ph_values = torch.clamp(ph_values, min=0.0, max=14.0)

            outputs = model(images, rgb_values)
            # Compute per-sample Huber loss
            sample_val_losses = criterion(outputs, ph_values).squeeze(1)  # shape [batch]
            # Take the mean over the batch for validation loss
            batch_val_loss = sample_val_losses.mean().item()
            val_loss += batch_val_loss

            # Log any unusually large sample losses
            batch_losses = (outputs - ph_values).abs()
            if (batch_losses > 10).any():
                print("⚠️ High residual in validation batch:", batch_losses.max().item())
    
    avg_val_loss = val_loss / len(val_loader)
    val_loss_history.append(avg_val_loss)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    print(f"Validation Loss: {avg_val_loss:.4f}")

    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        torch.save(model.state_dict(), best_model_path)
        with open("models/best_loss.txt", "w") as f:
            f.write(str(best_loss))
        print(f"✅ 新最佳模型已儲存，Validation Loss: {best_loss:.4f}")

    # Step scheduler
    scheduler.step(avg_val_loss)

    # Early stopping logic
    if avg_val_loss < best_loss:
        no_improve_epochs = 0
    else:
        no_improve_epochs += 1
        if no_improve_epochs >= early_stop_patience:
            print(f"🔻 Early stopping at epoch {epoch+1}")
            break

epochs_range = range(1, len(loss_history) + 1)
plt.figure()
plt.plot(epochs_range, loss_history, marker='o', label='Training Loss')
plt.plot(epochs_range, val_loss_history, marker='x', label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training & Validation Loss Curve")
plt.legend()
plt.grid(True)
plt.savefig("loss_curve.png")
plt.show()

## 不再重複儲存模型，僅保留訓練過程中損失最低的 best.pth

print("\n✅ 訓練完成")