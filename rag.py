import torch
import pandas as pd
import numpy as np
from PIL import Image
import cv2
from torchvision import transforms
from model import PHRegressionModel
#from dataset import rgb_str_to_tuple (將rgb值轉換為字串形式儲存)
from sklearn.metrics.pairwise import cosine_similarity
import heapq
import math


# Constants for model and data paths
MODEL_PATH = 'models/best.pth'
CSV_PATH = 'all_labels.csv'


# Device selection
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load model and database once at import time
_MODEL = None
_VECTOR_DB = None
_DB_VECTORS = None
_DB_PHS = None
def _initialize_globals():
    global _MODEL, _VECTOR_DB, _DB_VECTORS, _DB_PHS
    if _MODEL is None:
        _MODEL = load_model(MODEL_PATH, DEVICE)
    if _VECTOR_DB is None:
        _VECTOR_DB = build_vector_database(CSV_PATH)
        _DB_VECTORS = np.stack(_VECTOR_DB['rgb_vector'].values)
        _DB_PHS = _VECTOR_DB['ph_value'].values

# 載入回歸模型
def load_model(model_path='models/best.pth', device='cpu'):
    model = PHRegressionModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# 圖像轉 RGB + tensor（與訓練一致的預處理）
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    # Same preprocessing as training: resize, to tensor, normalize
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    tensor = transform(image).unsqueeze(0)
    # Compute average RGB in original scale (0–255)
    np_img = np.array(image)
    avg_rgb = np_img.mean(axis=(0, 1)).astype(int)
    rgb = (avg_rgb[2], avg_rgb[1], avg_rgb[0])
    return tensor, rgb

# 建立向量庫（過濾 testdrop）
def build_vector_database(csv_path):
    df = pd.read_csv(csv_path)
    df = df[df['ph_value'] != 'testdrop']
    df = df[df['ph_value'] != 0]
    df['ph_value'] = df['ph_value'].astype(float)
    df[['rgb_r', 'rgb_g', 'rgb_b']] = df[['rgb_r', 'rgb_g', 'rgb_b']].astype(float)
    df['rgb_vector'] = df[['rgb_r', 'rgb_g', 'rgb_b']].values.tolist()
    return df

def find_nearest_rgb_match(rgb_vector, vector_df, k=4, db_vectors=None, db_phs=None):
    """
    Find k nearest color matches by Euclidean distance in RGB space.
    Returns weighted average pH and an inverse-distance "similarity" score.
    """
    # Prepare target and database vectors
    target = np.array(rgb_vector, dtype=float)
    if db_vectors is None:
        vectors = np.stack(vector_df['rgb_vector'].values).astype(float)
    else:
        vectors = db_vectors.astype(float)

    # Compute Euclidean distances
    distances = np.linalg.norm(vectors - target, axis=1)

    # Find indices of k smallest distances
    top_k_indices = np.argsort(distances)[:k]
    top_k_distances = distances[top_k_indices]

    # Get corresponding pH values
    if db_phs is None:
        top_k_phs = vector_df.iloc[top_k_indices]['ph_value'].values.astype(float)
    else:
        top_k_phs = db_phs[top_k_indices].astype(float)

    # Compute weights: inverse distance (add small epsilon to avoid div by zero)
    eps = 1e-6
    weights = 1.0 / (top_k_distances + eps)

    # Weighted average pH
    weighted_ph = np.average(top_k_phs, weights=weights)

    # Similarity score: convert minimal distance to a bounded score
    min_dist = top_k_distances.min()
    similarity = 1.0 / (1.0 + min_dist)

    return weighted_ph, similarity

# 主預測函式（整合）
def predict_ph(image_path, sensor_rgb=None, model_path='models/best.pth', csv_path='all_labels.csv'):
    # Use pre-loaded globals
    model = _MODEL
    vector_db = _VECTOR_DB
    db_vectors = _DB_VECTORS
    db_phs = _DB_PHS
    device = DEVICE

    # 圖像處理
    image_tensor, image_rgb = preprocess_image(image_path)
    image_tensor = image_tensor.to(device)
    image_rgb_tensor = torch.tensor(image_rgb, dtype=torch.float32).unsqueeze(0).to(device)

    # 預測：模型內部同時融合影像與 RGB
    with torch.no_grad():
        avg_ph = model(image_tensor, image_rgb_tensor).item()

    # 圖像 RGB 比對
    image_rgb_ph, sim1 = find_nearest_rgb_match(image_rgb, vector_db, k=4, db_vectors=db_vectors, db_phs=db_phs)

    if sensor_rgb is not None:
        sensor_ph, sim2 = find_nearest_rgb_match(sensor_rgb, vector_db, k=4, db_vectors=db_vectors, db_phs=db_phs)
    else:
        sensor_ph, sim2 = None, None

    # Final predicted pH as average of model, image, and sensor predictions
    values = [avg_ph, image_rgb_ph]
    weights = [1.0, 1.0]  # equal weight for model and image by default
    if sensor_ph is not None:
        values.append(sensor_ph)
        weights.append(1.0)
    final_ph = float(np.average(values, weights=weights))

    # 結果整理
    result = {
        '回歸模型預測': round(avg_ph, 3),
        '圖像RGB比對': round(image_rgb_ph, 3),
        '圖像比對相似度': round(sim1, 3),
        '感測器RGB比對': round(sensor_ph, 3) if sensor_ph is not None else None,
        '感測器比對相似度': round(sim2, 3) if sim2 is not None else None,
        '最終預測pH值': round(final_ph, 3),
    }
    return result

# 範例使用
#
# Initialize model and database after all functions are defined
_initialize_globals()
if __name__ == '__main__':
    image_path = '/Users/ian/segmentation_prj/all_data/IMG_3987_circle_4_20250514_203045.jpg'
    sensor_rgb = (200, 180, 160)  # 模擬感測器 RGB 值

    result = predict_ph(image_path, sensor_rgb)
    print("\n📊 綜合分析結果：")
    for k, v in result.items():
        if v is not None:
            print(f"{k}: {v}")