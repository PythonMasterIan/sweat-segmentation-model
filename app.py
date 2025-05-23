from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
from rag import predict_ph
import os
import csv
import json
from werkzeug.utils import secure_filename
from PIL import Image
import io

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'upload_data')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
CORS(app)  # allow cross-origin requests

def get_next_index():
    # 支援 png 命名
    existing = [f for f in os.listdir(UPLOAD_FOLDER) if f.startswith("data") and f.endswith(".png")]
    indices = [int(f[4:-4]) for f in existing if f[4:-4].isdigit()]
    return max(indices, default=0) + 1

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    image_path = data.get('image_path')
    sensor_rgb = data.get('sensor_rgb')  # e.g. [200, 180, 160]
    result = predict_ph(
        image_path,
        sensor_rgb=tuple(sensor_rgb) if sensor_rgb else None
    )
    return jsonify(result)

@app.route('/upload-photo', methods=['POST'])
def upload_photo():
    print("📥 收到 POST /upload-photo 請求")
    print("🧾 Header 欄位：")
    for k, v in request.headers.items():
        print(f"  {k}: {v}")
    print("📦 資料長度：", len(request.get_data()))

    if not all(k in request.headers for k in ('r', 'g', 'b', 'width', 'height')):
        return jsonify({'status': 'error', 'message': '缺少必要欄位'}), 400

    try:
        r = int(request.headers.get('r', -1))
        g = int(request.headers.get('g', -1))
        b = int(request.headers.get('b', -1))
        width = int(request.headers.get('width', -1))
        height = int(request.headers.get('height', -1))
        if -1 in (r, g, b, width, height):
            raise ValueError("欄位缺失或為空")
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'欄位格式錯誤：{e}'}), 400

    raw_data = request.get_data()
    if len(raw_data) != width * height * 2:
        return jsonify({'status': 'error', 'message': f"raw buffer size mismatch: expected {width*height*2} bytes"}), 400

    # 轉換 RGB565 raw buffer 為 RGB 圖像
    pixels = []
    for i in range(0, len(raw_data), 2):
        byte1 = raw_data[i]
        byte2 = raw_data[i+1]
        value = (byte1 << 8) | byte2
        r5 = (value >> 11) & 0x1F
        g6 = (value >> 5) & 0x3F
        b5 = value & 0x1F
        rgb = (
            int(r5 * 255 / 31),
            int(g6 * 255 / 63),
            int(b5 * 255 / 31)
        )
        pixels.append(rgb)

    img = Image.new("RGB", (width, height))
    img.putdata(pixels)

    index = get_next_index()
    image_filename = f"data{index}.png"
    csv_filename = f"data{index}.csv"
    filepath = os.path.join(UPLOAD_FOLDER, secure_filename(image_filename))
    try:
        img.save(filepath)
        print(f"✅ 圖片儲存完成：{filepath}")
    except Exception as e:
        print(f"❌ 圖片儲存失敗：{e}")
        return jsonify({'status': 'error', 'message': f'圖片儲存失敗: {e}'}), 500

    csvpath = os.path.join(UPLOAD_FOLDER, secure_filename(csv_filename))
    with open(csvpath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['r', 'g', 'b'])
        writer.writerow([r, g, b])

    print(f"✅ 接收並儲存 data{index}.png, RGB: ({r}, {g}, {b})")

    print(f"📈 預測中：影像={filepath}, RGB=({r}, {g}, {b})")
    try:
        result = predict_ph(filepath, sensor_rgb=(r, g, b))
        print(f"📈 預測結果為：{result}")
        # 儲存 latest_result.json
        latest_result_path = os.path.join(UPLOAD_FOLDER, 'latest_result.json')
        with open(latest_result_path, 'w') as f:
            json.dump({'index': index, 'result': result}, f)
    except Exception as e:
        print(f"❌ 預測失敗：{e}")
        return jsonify({'status': 'error', 'message': f'預測失敗: {e}'}), 500

    return jsonify({'status': 'ok', 'index': index, 'result': result})


# 新增 /latest-result 路由
@app.route('/latest-result', methods=['GET'])
def latest_result():
    latest_path = os.path.join(UPLOAD_FOLDER, 'latest_result.json')
    if not os.path.exists(latest_path):
        return jsonify({'status': 'error', 'message': '尚無分析結果'}), 404
    with open(latest_path, 'r') as f:
        data = json.load(f)
    return jsonify(data)

@app.route('/trigger', methods=['GET'])
def trigger_response():
    return jsonify({'status': 'ok', 'message': '✅ 感測完成並上傳'})

# 新增靜態檔案路由
@app.route('/upload_data/<filename>')
def uploaded_file(filename):
    fullpath = os.path.join(UPLOAD_FOLDER, filename)
    print(f"🧾 嘗試讀取檔案：{fullpath}")
    if not os.path.exists(fullpath):
        print("❌ 找不到檔案")
        return "❌ 找不到檔案", 404
    return send_file(fullpath)

if __name__ == '__main__':
    # Listen on all network interfaces, port 5001
    app.run(host='0.0.0.0', port=5001, debug=True)