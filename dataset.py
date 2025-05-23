import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class PHImageRGBDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

        self.data = self.data[self.data['filename'].apply(lambda x: os.path.isfile(os.path.join(self.image_dir, x)))].reset_index(drop=True)

        # 移除 pH 為空或為非數字的資料（如 0）
        self.data = self.data[pd.to_numeric(self.data['ph_value'], errors='coerce').notnull()]
        self.data['ph_value'] = self.data['ph_value'].astype(float)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # 讀取圖像
        img_path = os.path.join(self.image_dir, row['filename'])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        # 讀取 RGB 特徵
        rgb = torch.tensor([row['rgb_r'], row['rgb_g'], row['rgb_b']], dtype=torch.float32)

        # 目標值：pH
        ph = torch.tensor(row['ph_value'], dtype=torch.float32)

        return image, rgb, ph