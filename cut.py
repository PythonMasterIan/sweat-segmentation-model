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
from tkinter import simpledialog, Tk, filedialog

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
    
    # 單張模式下不再初始化 all_labels.csv
    
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
    if circles is not None:
        circles = np.uint16(np.around(circles))

    # 永遠詢問是否補充手動標記
    count_detected = 0 if circles is None else len(circles[0])
    print(f"⚠️ 偵測到 {count_detected} 個圓形")

    # 詢問使用者是否要進入手動補充模式
    user_select_manual = simpledialog.askstring("補充標記", f"目前偵測到 {count_detected} 個圓形，是否要補充標記？(y/n)", parent=tk_root)
    if user_select_manual is None or user_select_manual.strip().lower() != "y":
        print("🔵 使用者選擇不進入手動補充模式")
        if circles is None:
            print("❌ 無有效圓形，結束")
            return None, None, None
        # 若有圓形，直接畫標記圖
        marked_image = original.copy()
        for i, (x, y, r) in enumerate(circles[0, :]):
            # 畫圓和中心點
            cv2.circle(marked_image, (x, y), r, (0, 255, 0), 2)
            cv2.circle(marked_image, (x, y), 2, (0, 0, 255), 3)
            # 標記編號 (加大字體、黑色)
            cv2.putText(marked_image, f"{i+1}", (x-20, y-r-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 5.0, (0, 0, 0), 10)
    else:
        print("🖱️ 使用者選擇進入手動標記模式")
        # 後面原本的手動補選程式保留，無需修改
        manual_circles = []
        from matplotlib.widgets import RectangleSelector

        # 若已有自動偵測結果，先畫出來（僅初始化底圖，不重繪circles）
        marked_image = original.copy()
        # 新增：畫出已偵測到的圓形（綠色框）
        if circles is not None:
            for i, (x, y, r) in enumerate(circles[0, :]):
                cv2.circle(marked_image, (x, y), r, (0, 255, 0), 2)
                cv2.circle(marked_image, (x, y), 2, (0, 0, 255), 3)
                cv2.putText(marked_image, f"{i+1}", (x-20, y-r-20),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 4)

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(cv2.cvtColor(marked_image, cv2.COLOR_BGR2RGB))
        ax.set_title("拖曳滑鼠以補充標記圓形（正圓）區域（ESC取消上次，關閉視窗結束）")
        plt.axis('off')

        # RectangleSelector callback (drawn rectangle converted to circle)
        def on_select(eclick, erelease):
            x1, y1 = eclick.xdata, eclick.ydata
            x2, y2 = erelease.xdata, erelease.ydata
            xc = int(round((x1 + x2) / 2.0))
            yc = int(round((y1 + y2) / 2.0))
            side = min(abs(x2 - x1), abs(y2 - y1)) / 2.0
            r = int(round(side))
            if r > 0:
                manual_circles.append((xc, yc, r))
                # Draw the circle
                cv2.circle(marked_image, (xc, yc), r, (255, 0, 0), 2)
                ax.cla()
                ax.imshow(cv2.cvtColor(marked_image, cv2.COLOR_BGR2RGB))
                ax.set_title("拖曳滑鼠以補充標記圓形（正圓）區域（ESC取消上次，關閉視窗結束）")
                plt.axis('off')
                plt.draw()
                print(f"🖱️ 拖曳座標: ({int(x1)}, {int(y1)}) 到 ({int(x2)}, {int(y2)})，中心: ({xc}, {yc}), 半徑: {r}")

        # 新增 ESC 取消選取功能
        def on_keypress(event):
            if event.key == 'escape' and manual_circles:
                print("↩️ 取消上一次選取的圓形")
                manual_circles.pop()  # 移除最後一個圓形
                marked_image = original.copy()
                # 只重繪 manual_circles，確保被取消的圓形不會再次顯示
                for x, y, r in manual_circles:
                    cv2.circle(marked_image, (x, y), r, (255, 0, 0), 2)

                ax.clear()
                ax.imshow(cv2.cvtColor(marked_image, cv2.COLOR_BGR2RGB))
                ax.set_title("拖曳滑鼠以補充標記圓形（正圓）區域（按ESC取消上次，關閉視窗結束）")
                plt.axis('off')
                plt.draw()

        rectangle_selector = RectangleSelector(
            ax,
            on_select,
            useblit=True,
            button=[1],  # left mouse
            minspanx=5, minspany=5,
            spancoords='pixels',
            interactive=True,
        )

        fig.canvas.mpl_connect('key_press_event', on_keypress)

        print("🔵 拖曳滑鼠於圖像上以圈選圓形，完成後請關閉視窗。")
        plt.show(block = True)

        # 完成後將手動畫的圓形加入circles
        if manual_circles:
            manual_arr = np.array([[x, y, r] for x, y, r in manual_circles], dtype=np.uint16).reshape(1, -1, 3)
            if circles is not None:
                circles = np.concatenate([circles, manual_arr], axis=1)
            else:
                circles = manual_arr
            print(f"✅ 使用者補充標記了 {len(manual_circles)} 個圓形，總計 {len(circles[0])} 個圓形")
            input("🔽 標記已完成，請按 Enter 繼續選擇要處理的圓形...")
        else:
            if circles is None:
                print("❌ 無法檢測或標記圓形，結束")
                return None, None, None
            else:
                print("🔵 未補充新標記，繼續使用已偵測的圓形")
    
    # 顯示標記後的圖像，讓使用者了解圓形編號
    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(marked_image, cv2.COLOR_BGR2RGB))
    plt.title(f'偵測到 {len(circles[0])} 個圓形')
    plt.axis('off')
    
    
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
            cv2.putText(result_image, f"{circle_idx+1}", (x-20, y-r-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 5.0, (0, 0, 0), 10)
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
        # 在同一個 figure 中用 subplot 顯示原圖和裁切圖
        fig = plt.figure(figsize=(18, 10))

        # 左側：原始標記圖像
        ax1 = plt.subplot(1, 2, 1)
        ax1.imshow(cv2.cvtColor(marked_image, cv2.COLOR_BGR2RGB))
        ax1.set_title('原始標記圖像')
        ax1.axis('off')

        # 右側：裁切圖像們，組合成一張網格圖像
        ax2 = plt.subplot(1, 2, 2)
        num = len(cropped_circles)
        grid_cols = 4
        grid_rows = (num + grid_cols - 1) // grid_cols
        thumb_h, thumb_w = 100, 100
        canvas = np.ones((grid_rows * thumb_h, grid_cols * thumb_w, 3), dtype=np.uint8) * 255

        for i, crop in enumerate(cropped_circles):
            if crop is not None and crop.size > 0 and not np.all(crop == 0):
                resized = cv2.resize(crop, (thumb_w, thumb_h))
            else:
                resized = np.zeros((thumb_h, thumb_w, 3), dtype=np.uint8)
                cv2.putText(resized, '無效', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            row = i // grid_cols
            col = i % grid_cols
            canvas[row*thumb_h:(row+1)*thumb_h, col*thumb_w:(col+1)*thumb_w] = resized

        ax2.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        ax2.set_title('裁切圓形預覽')
        ax2.axis('off')

        plt.tight_layout()
        plt.draw()
        print("\n>>> 正在顯示裁切結果，請參考圖中編號 <<<")
        plt.pause(0.1)
        
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
                    ph_value = (ph_value or "").strip()
                    if not ph_value:
                        ph_value = "unknown"
                    # 允許 "testdrop" 作為 ph_value
                    ph_values[i] = ph_value

            # 準備單一 CSV 檔
            csv_name = f"{base_filename}.csv"
            csv_path = os.path.join(output_path, csv_name)
            # 'w' 模式會覆蓋同名檔案，若存在則覆蓋
            with open(csv_path, 'w', encoding='utf-8', newline='') as fcsv:
                writer = csv.writer(fcsv)
                writer.writerow(['filename', 'ph_value', 'date_created', 'source_image', 'rgb_color'])

                for i in indices_to_save:
                    if 0 <= i < len(cropped_circles):
                        crop = cropped_circles[i]
                        orig_idx = original_indices[i]

                        if crop is not None and crop.size > 0 and not np.all(crop == 0):
                            # 生成唯一文件名
                            filename = f'{base_filename}_circle_{orig_idx+1}_{timestamp}.jpg'
                            save_path = os.path.join(output_path, filename)

                            # 保存圖像
                            cv2.imwrite(save_path, crop)
                            print(f"✅ 已保存圓形 {orig_idx+1} 到 {save_path}")

                            # 寫入同一 CSV，增加rgb_color欄位
                            writer.writerow([
                                filename,
                                ph_values[i],
                                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                os.path.basename(image_path),
                                str(circle_colors[i])  # RGB tuple as string
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
    print("\n" + "="*50)
    print("🔍 圓形區域檢測與pH值標記工具 🎨")
    print("="*50)

    last_dir_path = os.getcwd()

    while True:
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
            input_dir = filedialog.askdirectory(title="選擇包含圖像的目錄", initialdir=last_dir_path)
            if not input_dir or not os.path.isdir(input_dir):
                print(f"❌ 目錄不存在: {input_dir}")
                break
            last_dir_path = input_dir
            # 固定目標圓形數量為7
            target_count = 7
            batch_process(input_dir, target_count=target_count)
            break
        else:
            # 單張圖片處理模式
            while True:
                image_path = filedialog.askopenfilename(
                    title="選擇圖片檔案",
                    initialdir=last_dir_path,
                    filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
                )
                if not image_path or not os.path.isfile(image_path):
                    print(f"❌ 文件不存在: {image_path}")
                    # 詢問是否繼續
                    retry = simpledialog.askstring("未選擇檔案", "未選擇檔案，是否要結束？(y/n):", parent=tk_root)
                    if retry and retry.strip().lower() == "y":
                        sys.exit(0)
                    else:
                        continue
                last_dir_path = os.path.dirname(image_path)

                # 執行檢測和提取（使用固定參數）
                circles, colors, crops = extract_circle_colors(
                    image_path=image_path,
                    dp=1,
                    min_dist=150,
                    param1=100,
                    param2=47,
                    min_radius=200,
                    max_radius=250,
                    show_results=True,
                    save_results=True
                )

                # 顯示提取的顏色值
                if circles is not None and colors:
                    print("\n🎨 提取的顏色值:")
                    for i, color in enumerate(colors):
                        print(f"圓形 {i+1}: RGB = {color}")

                print("\n✨ 處理完成! ✨")
                # 處理完一張圖片自動跳出下次選擇
                # 直接重新進入下一輪選擇，不回到終端機
                # 若使用者取消選擇，詢問是否結束
                # 這個 while True 只會在用戶選擇結束時才 break