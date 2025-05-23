import os
import sys
import argparse
import numpy as np
import pandas as pd
from rag import predict_ph


def evaluate(test_csv, image_root, sensor_rgb=None):
    """
    Evaluate pH model accuracy on a test set.
    - test_csv: CSV file with columns ['image_filename', 'ph_value']
    - image_root: directory path where image files are stored
    - sensor_rgb: optional tuple of (R,G,B) from sensor
    """
    df = pd.read_csv(test_csv)
    # Remove test samples marked as 'testdrop' or with non-positive pH
    df = df[ (df['ph_value'] != 'testdrop') & (df['ph_value'].astype(float) > 0) ]
    # Determine filename column
    possible_cols = ['image_filename', 'filename', 'file_name', 'image']
    for col in possible_cols:
        if col in df.columns:
            filename_col = col
            break
    else:
        # Fallback to first column
        filename_col = df.columns[0]
    errors = []  # list to collect (error, filename, predicted, true)
    y_true = []
    y_pred = []

    for _, row in df.iterrows():
        image_path = os.path.join(image_root, row[filename_col])
        if not os.path.isfile(image_path):
            print(f"⚠️ 檔案不存在: {image_path}", file=sys.stderr)
            continue
        result = predict_ph(image_path, sensor_rgb)
        y_pred.append(result['回歸模型預測'])
        y_true.append(float(row['ph_value']))
        # record error for this sample
        true_val = float(row['ph_value'])
        pred_val = result['回歸模型預測']
        errors.append((abs(pred_val - true_val), row[filename_col], pred_val, true_val))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mae = np.mean(np.abs(y_pred - y_true))
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    corr = np.corrcoef(y_true, y_pred)[0, 1]

    # print top 10 highest error samples
    errors.sort(reverse=True, key=lambda x: x[0])
    print("Top 10 errors (absolute error, file, pred, true):")
    for err, fname, pred_val, true_val in errors[:10]:
        print(f"{err:.3f}, {fname}, pred={pred_val:.2f}, true={true_val:.2f}")

    print(f"Evaluated {len(y_true)} samples")
    print(f"MAE: {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"Correlation: {corr:.3f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate pH model accuracy")
    parser.add_argument(
        '--test_csv', type=str, default='all_labels.csv',
        help="Path to CSV with columns ['image_filename','ph_value']")
    parser.add_argument(
        '--image_root', type=str, default='all_data',
        help="Folder containing the test images")
    parser.add_argument(
        '--sensor_rgb', nargs=3, type=int,
        help="Optional sensor RGB values, e.g. --sensor_rgb 200 180 160")

    args = parser.parse_args()
    sensor = tuple(args.sensor_rgb) if args.sensor_rgb else None
    evaluate(args.test_csv, args.image_root, sensor)