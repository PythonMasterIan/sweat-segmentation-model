import matplotlib
matplotlib.use("TkAgg") 
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import datetime
import csv
import sys
from tkinter import simpledialog, Tk

# 初始化 tkinter
tk_root = Tk()
tk_root.withdraw()

def extract_circle_colors(image_path, output_dir='output', 
                         dp=1, min_dist=150, param1=100, param2=47, 
                         min_radius=200, max_radius=250,
                         show_results=True, save_results=True):
    """
    檢測並提取圖像中圓形區域的顏色，並支援標籤註記
    
    參數:
        image_path: 輸入圖像路徑
        output_dir: 輸出目錄
        dp, min_dist, param1, param2, min_radius, max_radius: 圓形檢測參數
        show_results: 是否顯示結果
        save_results: 是否保存結果
    """
    start_time = time.time()  # 開始計時
    
    # 所有小圖統一儲存到 all_data/
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    output_path = "all_data"
    
    # 確保輸出目錄存在
    os.makedirs(output_path, exist_ok=True)
    print(f"✅ 使用輸出目錄: {output_path}")
    
    # 初始化 all_labels.csv 檔案（如果不存在就建立並寫入表頭）
    label_csv_path = "all_labels.csv"
    csv_exists = os.path.exists(label_csv_path)
    
    if not csv_exists:
        with open(label_csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['filename', 'ph_value', 'date_created', 'source_image'])
        print(f"✅ 創建標籤文件: {label_csv_path}")
    else:
        print(f"✅ 使用現有標籤文件: {label_csv_path}")
    
    # 載入圖像
    print(f"🔍 正在載入圖像：{image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ 無法載入圖像：{image_path}")
        return None, None, None
    
    # 保存原始圖像用於顯示和保存結果
    original = image.copy()
    
    # 轉換為RGB（用於顯示）和灰度圖像（用於檢測）
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 圖像預處理：適當模糊以去除噪點
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    print(f"✅ 圖像載入成功: {image.shape[1]}x{image.shape[0]} 像素")
    
    # 檢測圓形
    print(f"⚙️ 使用參數: dp={dp}, minDist={min_dist}, param1={param1}, param2={param2}, minRadius={min_radius}, maxRadius={max_radius}")
    
    try:
        # 使用HoughCircles檢測圓形
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=dp,
            minDist=min_dist,
            param1=param1,
            param2=param2,
            minRadius=min_radius,
            maxRadius=max_radius
        )
        
        if circles is None:
            print("⚠️ 沒有找到任何圓形，嘗試調整參數...")
            
            # 嘗試更寬鬆的參數
            circles = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                dp=dp,
                minDist=min_dist//2,  # 減少最小距離
                param1=param1,
                param2=param2//2,     # 降低檢測閾值
                minRadius=min_radius-10,
                maxRadius=max_radius+10
            )
            
            if circles is None:
                print("❌ 依然無法檢測到圓形，請手動調整參數")
                return None, None, None
            else:
                print(f"✅ 使用調整後的參數成功檢測到圓形")
    except Exception as e:
        print(f"❌ 圓形檢測時出錯: {str(e)}")
        return None, None, None
    
    # 轉換為整數坐標
    circles = np.uint16(np.around(circles))
    
    # 已註解掉的過濾重疊圓形區塊已安全移除
    
    print(f"✅ 偵測到 {len(circles[0])} 個圓形")
    
    # 在原始圖像上標記檢測到的圓形
    marked_image = original.copy()
    for i, (x, y, r) in enumerate(circles[0, :]):
        # 畫圓和中心點
        cv2.circle(marked_image, (x, y), r, (0, 255, 0), 2)
        cv2.circle(marked_image, (x, y), 2, (0, 0, 255), 3)
        # 標記編號
        cv2.putText(marked_image, f"{i+1}", (x-10, y-r-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    
    # 顯示標記後的圖像，讓使用者了解圓形編號
    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(marked_image, cv2.COLOR_BGR2RGB))
    plt.title(f'偵測到 {len(circles[0])} 個圓形')
    plt.axis('off')
    
    # 保存初始標記圖像，以便之後參考
    initial_marked_image_path = os.path.join(output_path, f'{base_filename}_marked.jpg')
    cv2.imwrite(initial_marked_image_path, marked_image)
    
    # 顯示圖像並等待用戶輸入
    plt.draw()
    print("\n>>> 正在顯示圓形檢測結果，請參考圖中編號 <<<")
    plt.pause(0.1)  # 必要的暫停，以確保GUI更新
    
    # 第一步選擇：選擇要處理的圓形
    process_indices_input = simpledialog.askstring("選擇圓形", "\n請輸入要處理的圓形編號（以逗號分隔，例如：1,3,5），或按Enter處理所有：", parent=tk_root)
    if process_indices_input is None:
        process_indices_input = ""
    
    if process_indices_input.strip():
        # 轉換為0-based索引
        process_indices = []
        for idx in process_indices_input.split(','):
            if idx.strip().isdigit():
                idx_num = int(idx.strip()) - 1  # 轉為0-based
                if 0 <= idx_num < len(circles[0]):
                    process_indices.append(idx_num)
                else:
                    print(f"⚠️ 忽略無效的編號 {int(idx.strip())}")
        
        if not process_indices:
            print("⚠️ 未提供有效的編號，將處理所有圓形")
            circles_to_process = circles[0]
            process_indices = list(range(len(circles[0])))
        else:
            circles_to_process = np.array([circles[0][i] for i in process_indices])
            print(f"✅ 將處理 {len(circles_to_process)} 個選中的圓形")
    else:
        # 處理所有圓形
        circles_to_process = circles[0]
        process_indices = list(range(len(circles[0])))
        print(f"✅ 將處理所有 {len(circles_to_process)} 個圓形")
    
    # 關閉初始圖像窗口
    plt.close()
    
    # 準備結果容器
    circle_colors = []
    cropped_circles = []
    # 記錄原始索引，用於後續顯示和保存
    original_indices = []
    
    # 創建用於顯示的圖像
    result_image = original.copy()
    
    # 處理選中的圓形
    for i, circle_idx in enumerate(process_indices):
        x, y, r = circles[0, circle_idx]
        original_indices.append(circle_idx)

        # 在結果圖像上畫圓
        cv2.circle(result_image, (x, y), r, (0, 255, 0), 2)
        cv2.circle(result_image, (x, y), 2, (0, 0, 255), 3)

        # 創建圓形掩碼
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.circle(mask, (x, y), r-2, 255, -1)  # r-2避免包含邊緣

        # 應用掩碼
        masked_data = cv2.bitwise_and(image, image, mask=mask)

        # 提取非黑色像素用於計算平均顏色
        mask_pixels = np.where(mask > 0)
        if len(mask_pixels[0]) > 0:
            # 獲取有效像素
            valid_pixels = image[mask_pixels]
            # 計算平均顏色 (BGR順序)
            avg_color = np.mean(valid_pixels, axis=0).astype(int)
            # 轉換為RGB順序用於顯示
            avg_color_rgb = (int(avg_color[2]), int(avg_color[1]), int(avg_color[0]))
            circle_colors.append(avg_color_rgb)
            # 在結果圖像上標記編號和顏色值
            cv2.putText(result_image, f"{circle_idx+1}", (x-10, y-r-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            # 繪製顏色方塊
            cv2.rectangle(result_image, (x+r+5, y-15), (x+r+35, y+15),
                          (int(avg_color[0]), int(avg_color[1]), int(avg_color[2])), -1)
            cv2.rectangle(result_image, (x+r+5, y-15), (x+r+35, y+15),
                          (255, 255, 255), 1)
        else:
            circle_colors.append((0, 0, 0))

        # 裁切圓形區域
        x_min, y_min = max(0, x-r), max(0, y-r)
        x_max, y_max = min(image.shape[1], x+r), min(image.shape[0], y+r)
        # 安全裁切
        try:
            if x_max > x_min and y_max > y_min:
                cropped = masked_data[y_min:y_max, x_min:x_max]
                if cropped.size > 0:
                    cropped_circles.append(cropped)
                else:
                    print(f"⚠️ 圓形 {circle_idx+1} 裁切後為空")
                    cropped_circles.append(np.zeros((10, 10, 3), dtype=np.uint8))
            else:
                print(f"⚠️ 圓形 {circle_idx+1} 裁切區域無效")
                cropped_circles.append(np.zeros((10, 10, 3), dtype=np.uint8))
        except Exception as e:
            print(f"⚠️ 處理圓形 {circle_idx+1} 時出錯: {str(e)}")
            cropped_circles.append(np.zeros((10, 10, 3), dtype=np.uint8))
    
    # 顯示裁切的圓形與原始標記圖像並排顯示
    if cropped_circles:
        # 創建一個新的大圖，包含原始標記圖像和裁切圓形
        fig = plt.figure(figsize=(18, 10))
        
        # 左側：顯示原始標記圖像
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(marked_image, cv2.COLOR_BGR2RGB))
        plt.title('原始標記圖像')
        plt.axis('off')
        
        # 右側：顯示裁切的圓形
        plt.subplot(1, 2, 2)
        rows = (len(cropped_circles) + 3) // 4
        grid_fig = plt.figure(figsize=(12, 3 * rows))
        
        for i, crop in enumerate(cropped_circles):
            plt.subplot(rows, 4, i+1)
            try:
                if crop is not None and crop.size > 0 and not np.all(crop == 0):
                    plt.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                else:
                    plt.text(0.5, 0.5, '無效區域', ha='center', va='center')
            except:
                plt.text(0.5, 0.5, '顯示錯誤', ha='center', va='center')
            plt.title(f'圓形 {original_indices[i]+1}')
            plt.axis('off')
        
        plt.tight_layout()
        
        # 顯示圖像
        plt.draw()
        print("\n>>> 正在顯示裁切結果，請參考圖中編號 <<<")
        plt.pause(0.1)  # 必要的暫停，以確保GUI更新
        
        # 詢問要保存哪些圓形
        save_indices_input = simpledialog.askstring("保存圓形", "\n請輸入要保存的圓形編號（以逗號分隔，例如：1,3,5），或按Enter保存所有處理過的圓形：", parent=tk_root)
        if save_indices_input is None:
            save_indices_input = ""
        
        indices_to_save = []
        if save_indices_input.strip():
            # 解析使用者輸入的編號
            for idx in save_indices_input.split(','):
                if idx.strip().isdigit():
                    idx_num = int(idx.strip())
                    # 檢查是否為有效的圓形編號
                    for i, orig_idx in enumerate(original_indices):
                        if orig_idx + 1 == idx_num:  # +1轉為1-based
                            indices_to_save.append(i)
                            break
                    else:
                        print(f"⚠️ 忽略未處理的編號 {idx_num}")
        else:
            # 保存所有處理過的圓形
            indices_to_save = list(range(len(cropped_circles)))
        
        # 保存選中的圓形 (保持圖像窗口開啟)
        if indices_to_save:
            saved_count = 0
            saved_indices = []
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 收集所有 pH 值，允許 "testdrop" 關鍵字
            ph_values = {}
            for i in indices_to_save:
                if 0 <= i < len(cropped_circles):
                    orig_idx = original_indices[i]
                    ph_value = simpledialog.askstring(
                        "輸入pH值",
                        f"請輸入圓形 {orig_idx+1} 的 pH 值（可為小數，可輸入 'testdrop' 表示測試液）：",
                        parent=tk_root
                    )
                    if ph_value is None:
                        ph_value = ""
                    ph_value = ph_value.strip()
                    # 允許 "testdrop" 作為 ph_value
                    ph_values[i] = ph_value
            
            # 保存所有圖像和更新 CSV
            with open(label_csv_path, 'a', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                
                for i in indices_to_save:
                    if 0 <= i < len(cropped_circles):
                        crop = cropped_circles[i]
                        orig_idx = original_indices[i]
                        
                        if crop is not None and crop.size > 0 and not np.all(crop == 0):
                            # 生成唯一文件名（包含原始圖片名、圓形編號和時間戳）
                            filename = f'{base_filename}_circle_{orig_idx+1}_{timestamp}.jpg'
                            save_path = os.path.join(output_path, filename)
                            
                            # 保存圖像
                            cv2.imwrite(save_path, crop)
                            print(f"✅ 已保存圓形 {orig_idx+1} 到 {save_path}")
                            
                            # 寫入 CSV
                            writer.writerow([
                                filename, 
                                ph_values[i],
                                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                os.path.basename(image_path)
                            ])
                            
                            saved_count += 1
                            saved_indices.append(orig_idx)
            print(f"\n✨ 總共成功保存了 {saved_count} 個圓形及其 pH 值到 {output_path}")
        else:
            print("❌ 未選擇任何圓形進行保存")
        
        # 關閉圖像窗口
        plt.close('all')
    else:
        print("❌ 沒有有效的裁切圓形可以顯示")
    
    # 如果需要顯示結果
    if show_results:
        plt.figure(figsize=(12, 10))
        # 顯示處理後的圖像
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        plt.title(f'偵測到 {len(circles[0])} 個圓形')
        plt.axis('off')
        # 顯示顏色樣本
        plt.subplot(1, 2, 2)
        if circle_colors:
            color_sample_img = np.zeros((100, 100 * len(circle_colors), 3), dtype=np.uint8)
            for i, color in enumerate(circle_colors):
                # RGB轉回BGR用於OpenCV
                bgr_color = (color[2], color[1], color[0])
                color_sample_img[:, i*100:(i+1)*100] = bgr_color
                # 添加編號
                cv2.putText(color_sample_img, str(original_indices[i]+1), (i*100+40, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, 
                           (255, 255, 255) if sum(color) < 380 else (0, 0, 0), 
                           2)
            plt.imshow(cv2.cvtColor(color_sample_img, cv2.COLOR_BGR2RGB))
            plt.title('提取的顏色樣本')
        else:
            plt.text(0.5, 0.5, '沒有有效的顏色樣本', ha='center', va='center')
        plt.axis('off')
        plt.tight_layout()
        print("\n>>> 請查看最終結果，關閉窗口後結束程式 <<<")
        plt.show()
    
    # 計算並顯示處理時間
    end_time = time.time()
    print(f"\n⏱️ 總處理時間: {end_time - start_time:.2f} 秒")
    
    return circles[0], circle_colors, cropped_circles

# 設定參數自動搜索功能
def auto_tune_parameters(image_path, target_count=7, base_min_radius=15, base_max_radius=60):
    """
    自動調整參數尋找最佳圓形檢測結果
    
    參數:
        image_path: 輸入圖像路徑
        target_count: 目標圓形數量 
        base_min_radius: 基礎最小半徑
        base_max_radius: 基礎最大半徑
        
    返回:
        最佳參數設置
    """
    print("🔍 開始自動調整參數...")
    start_time = time.time()
    
    # 載入圖像
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ 無法載入圖像：{image_path}")
        return None
    
    # 轉換為灰度並模糊
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 初步估計圓形大小
    height, width = gray.shape
    min_dimension = min(height, width)
    
    # 根據圖像大小動態調整半徑範圍
    estimated_radius = min_dimension / 15  # 假設圖像中的圓形約佔圖像的1/15
    
    # 調整半徑搜索範圍
    if base_min_radius == 15 and base_max_radius == 60:  # 如果使用默認值，則進行動態調整
        base_min_radius = max(15, int(estimated_radius * 0.6))
        base_max_radius = min(int(estimated_radius * 1.4), min_dimension // 2)
        print(f"⚙️ 基於圖像大小調整半徑搜索範圍: {base_min_radius}-{base_max_radius}")
    
    # 嘗試不同參數組合
    best_params = None
    best_circle_count = 0
    
    # 參數範圍
    dp_values = [1.0, 1.2, 1.5]
    min_dist_values = [int(base_min_radius*1.5), int(base_min_radius*2), int(base_min_radius*3)]
    param1_values = [80, 100, 120]
    param2_values = [20, 30, 40, 50]
    
    total_combinations = len(dp_values) * len(min_dist_values) * len(param1_values) * len(param2_values)
    current_combination = 0
    
    print(f"🔄 將嘗試 {total_combinations} 種參數組合...")
    
    for dp in dp_values:
        for min_dist in min_dist_values:
            for param1 in param1_values:
                for param2 in param2_values:
                    current_combination += 1
                    
                    # 進度顯示
                    progress = (current_combination / total_combinations) * 100
                    sys.stdout.write(f"\r⏳ 進度: {progress:.1f}% [{current_combination}/{total_combinations}]")
                    sys.stdout.flush()
                    
                    # 嘗試當前參數組合
                    try:
                        circles = cv2.HoughCircles(
                            blurred,
                            cv2.HOUGH_GRADIENT,
                            dp=dp,
                            minDist=min_dist,
                            param1=param1,
                            param2=param2,
                            minRadius=base_min_radius,
                            maxRadius=base_max_radius
                        )
                        
                        # 如果找到圓形
                        if circles is not None:
                            circle_count = len(circles[0])
                            
                            # 如果找到目標數量的圓形，就是我們想要的結果
                            if circle_count == target_count:
                                print(f"\n✅ 找到最佳參數: dp={dp}, minDist={min_dist}, param1={param1}, param2={param2}")
                                return {
                                    'dp': dp,
                                    'min_dist': min_dist,
                                    'param1': param1,
                                    'param2': param2,
                                    'min_radius': base_min_radius,
                                    'max_radius': base_max_radius
                                }
                            
                            # 儲存最接近目標的結果
                            if best_params is None or abs(circle_count - target_count) < abs(best_circle_count - target_count):
                                best_params = {
                                    'dp': dp,
                                    'min_dist': min_dist,
                                    'param1': param1,
                                    'param2': param2,
                                    'min_radius': base_min_radius,
                                    'max_radius': base_max_radius
                                }
                                best_circle_count = circle_count
                    except Exception as e:
                        # 某些參數組合可能會導致錯誤，忽略它們
                        continue
    
    print()  # 換行，完成進度顯示
    
    # 如果沒有找到目標數量的圓形，返回最接近的結果
    if best_params:
        end_time = time.time()
        print(f"⚠️ 未找到正好{target_count}個圓形，最接近的參數檢測到 {best_circle_count} 個圓形")
        print(f"⏱️ 參數調整用時: {end_time - start_time:.2f} 秒")
        return best_params
    else:
        print("❌ 無法找到合適的參數")
        return None

def batch_process(input_dir, output_dir="all_data", target_count=7):
    """
    批量處理目錄中的圖像
    
    參數:
        input_dir: 輸入圖像目錄
        output_dir: 輸出目錄
        target_count: 每張圖像中期望的圓形數量
    """
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)]
    
    if not image_files:
        print(f"❌ 在 {input_dir} 中未找到有效的圖像文件")
        return
    
    print(f"🔍 找到 {len(image_files)} 個圖像文件")
    
    # 確保輸出目錄存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 處理每個圖像
    for i, image_file in enumerate(image_files):
        print(f"\n[{i+1}/{len(image_files)}] 處理圖像: {image_file}")
        image_path = os.path.join(input_dir, image_file)
        
        # 自動調整參數
        params = auto_tune_parameters(image_path, target_count=target_count)
        
        if params:
            # 使用調整後的參數處理圖像
            extract_circle_colors(
                image_path=image_path,
                output_dir=output_dir,
                dp=params['dp'],
                min_dist=params['min_dist'],
                param1=params['param1'],
                param2=params['param2'],
                min_radius=params['min_radius'],
                max_radius=params['max_radius'],
                show_results=True,
                save_results=True
            )
    
    print(f"\n✨ 批處理完成! 所有結果已保存到 {output_dir}")

if __name__ == "__main__":
    # 設置參數
    print("\n" + "="*50)
    print("🔍 圓形區域檢測與pH值標記工具 🎨")
    print("="*50)
    
    # 選擇功能模式
    print("\n請選擇操作模式:")
    print("1. 處理單張圖片")
    print("2. 批量處理目錄")
    mode = simpledialog.askstring("選擇模式", "請選擇操作模式 (1:單張圖片, 2:批量處理目錄):", parent=tk_root)
    if mode is None:
        mode = "1"
    mode = mode.strip()
    
    if mode == "2":
        # 批量處理模式
        input_dir = simpledialog.askstring("輸入目錄", "請輸入包含圖像的目錄路徑:", parent=tk_root)
        if input_dir is None or not os.path.isdir(input_dir):
            print(f"❌ 目錄不存在: {input_dir}")
            exit(1)
        
        target_count_input = simpledialog.askstring("期望圓形數量", "請輸入期望每張圖像中的圓形數量 (默認7):", parent=tk_root)
        if target_count_input is None or not target_count_input.strip():
            target_count = 7
        else:
            target_count = int(target_count_input)
        batch_process(input_dir, target_count=target_count)
    else:
        # 單張圖片處理模式
        image_path = simpledialog.askstring("圖像路徑", "請輸入圖像路徑:", parent=tk_root)
        if image_path is None or not os.path.isfile(image_path):
            print(f"❌ 文件不存在: {image_path}")
            exit(1)
        
        use_auto_tune_input = simpledialog.askstring("自動調參", "是否使用自動調參功能? (y/n):", parent=tk_root)
        use_auto_tune = (use_auto_tune_input or "").lower() == 'y'
        
        if use_auto_tune:
            # 自動調整參數
            target_count_input = simpledialog.askstring("期望圓形數量", "請輸入期望的圓形數量 (默認7):", parent=tk_root)
            if target_count_input is None or not target_count_input.strip():
                target_count = 7
            else:
                target_count = int(target_count_input)
            best_params = auto_tune_parameters(image_path, target_count=target_count)
            
            if best_params:
                # 使用最佳參數執行檢測和提取
                circles, colors, crops = extract_circle_colors(
                    image_path=image_path,
                    dp=best_params['dp'],
                    min_dist=best_params['min_dist'],
                    param1=best_params['param1'],
                    param2=best_params['param2'],
                    min_radius=best_params['min_radius'],
                    max_radius=best_params['max_radius'],
                    show_results=True,
                    save_results=True
                )
        else:
            # 使用手動設置的參數
            dp_input = simpledialog.askstring("dp參數", "輸入dp參數 (建議1.0-2.0):", parent=tk_root)
            dp = float(dp_input) if dp_input and dp_input.strip() else 1
            min_dist_input = simpledialog.askstring("圓心最小距離", "輸入圓心最小距離 (建議30-150):", parent=tk_root)
            min_dist = int(min_dist_input) if min_dist_input and min_dist_input.strip() else 150
            param1_input = simpledialog.askstring("param1", "輸入param1參數 (建議80-120):", parent=tk_root)
            param1 = int(param1_input) if param1_input and param1_input.strip() else 100
            param2_input = simpledialog.askstring("param2", "輸入param2參數 (建議20-50):", parent=tk_root)
            param2 = int(param2_input) if param2_input and param2_input.strip() else 47
            min_radius_input = simpledialog.askstring("最小半徑", "輸入最小半徑 (像素):", parent=tk_root)
            min_radius = int(min_radius_input) if min_radius_input and min_radius_input.strip() else 200
            max_radius_input = simpledialog.askstring("最大半徑", "輸入最大半徑 (像素):", parent=tk_root)
            max_radius = int(max_radius_input) if max_radius_input and max_radius_input.strip() else 250
            
            # 執行檢測和提取
            circles, colors, crops = extract_circle_colors(
                image_path=image_path,
                dp=dp,
                min_dist=min_dist,
                param1=param1,
                param2=param2,
                min_radius=min_radius,
                max_radius=max_radius,
                show_results=True,
                save_results=True
            )
        
        # 顯示提取的顏色值
        if circles is not None and colors:
            print("\n🎨 提取的顏色值:")
            for i, color in enumerate(colors):
                print(f"圓形 {i+1}: RGB = {color}")
        
        print("\n✨ 處理完成! ✨")