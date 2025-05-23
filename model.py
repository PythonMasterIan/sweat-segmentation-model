import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class PHRegressionModel(nn.Module):
    def __init__(self):
        super(PHRegressionModel, self).__init__()
        # 載入預訓練的 EfficientNet-B0 backbone
        self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        num_ftrs = self.backbone.classifier[1].in_features
        # 移除分類器以獲取特徵向量
        self.backbone.classifier = nn.Identity()

        # RGB 向量處理頭
        self.rgb_fc = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # 影像與 RGB 特徵融合後的回歸層
        self.fusion_fc = nn.Sequential(
            nn.Linear(num_ftrs + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, x_image, x_rgb):
        x_feat = self.backbone(x_image)  # 取出影像特徵
        x_rgb_feat = self.rgb_fc(x_rgb)  # 取出RGB特徵
        x_combined = torch.cat((x_feat, x_rgb_feat), dim=1)  # 合併特徵
        out = self.fusion_fc(x_combined)  # 融合後回歸
        return out