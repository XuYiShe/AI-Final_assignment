import cv2
from ultralytics import YOLO
from collections import defaultdict
import numpy as np # 确保导入

# --- 全局变量用于线条选择 ---
line_position = None  # 存储线的位置
line_orientation = None  # 'h' 代表水平线, 'v' 代表垂直线
selecting_line = True  # 用于控制是否正在选择线

# --- 鼠标回调函数，用于选择线 ---
def select_line(event, x, y, flags, param):
    global line_position, selecting_line
    
    if event == cv2.EVENT_LBUTTONDOWN and selecting_line:
        line_position = (x, y)
        selecting_line = False
        print(f"选择了{'水平' if line_orientation == 'h' else '垂直'}线: 位置 = {line_position}\n")

# --- 加载模型、视频、设置变量等 ---
model = YOLO('yolov8n.pt')
video_path = "inner_2min.mp4" # <--- 确保这里是你的视频路径
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"错误: 无法打开视频文件 {video_path}")
    exit()

track_history = defaultdict(lambda: [])
counted_ids = set() # 用于存储已经计数的ID
# --- 确保 line_y 的设置适合你的视频 ---
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

line_y = int(frame_height / 2) # 默认中间线
line_x = int(frame_width / 2)

ret, first_frame = cap.read()
if not ret:
    print("无法读取视频的第一帧")
    exit()

# --- 定义选择方向的窗口 ---
cv2.namedWindow("Select Line Type")
# 创建一个副本，以便在上面绘制文本
direction_selection = first_frame.copy()
cv2.putText(direction_selection, "Press 'h' for horizontal line", (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
cv2.putText(direction_selection, "Press 'v' for vertical line", (10, 60), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

cv2.imshow("Select Line Type", direction_selection)

# --- 等待用户选择方向 ---
while True:
    key = cv2.waitKey(0)
    if key == ord('h'):
        line_orientation = 'h'
        break
    elif key == ord('v'):
        line_orientation = 'v'
        break

# --- 让用户点击选择线的位置 ---
cv2.destroyWindow("Select Line Type")
cv2.namedWindow("Select Line Position")
cv2.setMouseCallback("Select Line Position", select_line)

# 根据不同方向显示提示
line_selection = first_frame.copy()
if line_orientation == 'h':
    cv2.putText(line_selection, "Click to set horizontal line position", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
else:
    cv2.putText(line_selection, "Click to set vertical line position", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

cv2.imshow("Select Line Position", line_selection)

# 等待用户点击选择
while selecting_line:
    cv2.waitKey(1)

# 销毁选择窗口
cv2.destroyWindow("Select Line Position")

# --- 根据用户选择设置线的位置 ---
if line_orientation == 'h':
    line_y = line_position[1]
else:
    line_x = line_position[0]

# --- 重新打开视频以从头开始 ---
cap.release()
cap = cv2.VideoCapture(video_path)

# --- 计数逻辑 ---
left_to_right_count = 0
right_to_left_count = 0
up_to_down_count = 0
down_to_up_count = 0
# ---

while True:
    _, frame = cap.read()

    # 视频结束或读取错误
    if frame is None:
        print("视频处理完成或读取帧出错 (frame is None).\n")
        break

    # --- 检测与跟踪 ---
    try:
        results = model.track(frame, persist=True, classes=[2], tracker='bytetrack.yaml', verbose=False) # 只跟踪汽车

        # --- 处理跟踪结果并绘制 ---
        annotated_frame = frame.copy() # 先复制原始帧，以防没有检测结果
        if results[0].boxes is not None and results[0].boxes.id is not None:
            # 使用 Ultralytics 内置绘图功能绘制框和ID
            annotated_frame = results[0].plot()

            # 获取框和ID用于计数逻辑 (如果需要可以从 results[0].boxes 获取)
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            # --- 计数逻辑 ---
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                center_y = int(y)
                center_x = int(x)
                track = track_history[track_id]
                track.append((float(x), float(y)))
                if len(track) > 15: track.pop(0)
                
                # 修改判断逻辑，增加更多线段检查点
                if track_id not in counted_ids and len(track) > 1:
                    # 检查连续两帧之间的多个点，而不仅仅是终点
                    for i in range(max(0, len(track)-5), len(track)-1):
                        prev_x, prev_y = track[i]
                        curr_x, curr_y = track[i+1]
                        
                        # 水平线判断（使用线性插值检查中间点）
                        if line_orientation == 'h':
                            # 如果两点分别在线的两侧
                            if (prev_y < line_y and curr_y >= line_y) or (prev_y >= line_y and curr_y < line_y):
                                # 计算交叉点
                                ratio = (line_y - prev_y) / (curr_y - prev_y) if curr_y != prev_y else 0
                                cross_x = prev_x + ratio * (curr_x - prev_x)
                                
                                # 判断方向并计数
                                if prev_y < line_y and curr_y >= line_y:
                                    down_to_up_count += 1
                                    counted_ids.add(track_id)
                                    cv2.circle(annotated_frame, (int(cross_x), int(line_y)), 5, (0, 255, 255), -1)
                                    break
                                elif prev_y >= line_y and curr_y < line_y:
                                    up_to_down_count += 1
                                    counted_ids.add(track_id)
                                    cv2.circle(annotated_frame, (int(cross_x), int(line_y)), 5, (255, 0, 0), -1)
                                    break
                        
                        # 垂直线判断
                        elif line_orientation == 'v':
                            # 如果两点分别在线的两侧
                            if (prev_x < line_x and curr_x >= line_x) or (prev_x >= line_x and curr_x < line_x):
                                # 计算交叉点
                                ratio = (line_x - prev_x) / (curr_x - prev_x) if curr_x != prev_x else 0
                                cross_y = prev_y + ratio * (curr_y - prev_y)
                                
                                # 判断方向并计数
                                if prev_x < line_x and curr_x >= line_x:
                                    left_to_right_count += 1
                                    counted_ids.add(track_id)
                                    cv2.circle(annotated_frame, (int(line_x), int(cross_y)), 5, (0, 255, 255), -1)
                                    break
                                elif prev_x >= line_x and curr_x < line_x:
                                    right_to_left_count += 1
                                    counted_ids.add(track_id)
                                    cv2.circle(annotated_frame, (int(line_x), int(cross_y)), 5, (255, 0, 0), -1)
                                    break
        # --- 在最终的 annotated_frame 上绘制计数线和总数 ---
        # 如果选择了水平线，绘制水平线
        if line_orientation == 'h':
            cv2.line(annotated_frame, (0, line_y), (frame_width, line_y), (0, 255, 0), 2)
        # 如果选择了垂直线，绘制垂直线
        if line_orientation == 'v':
            cv2.line(annotated_frame, (line_x, 0), (line_x, frame_height), (0, 255, 0), 2)

        #如果选择了水平线，只显示上下方向计数
        if line_orientation == 'h':
            cv2.putText(annotated_frame, f"Up to Down: {up_to_down_count}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(annotated_frame, f"Down to Up: {down_to_up_count}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        #如果选择了垂直线，只显示左右方向计数
        if line_orientation == 'v':
            cv2.putText(annotated_frame, f"Left to Right: {left_to_right_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(annotated_frame, f"Right to Left: {right_to_left_count}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        # --- >>>>> 关键：显示处理后的帧 <<<<< ---
        cv2.imshow("YOLOv8 Car Tracking & Counting", annotated_frame)

        # --- >>>>> 关键：等待按键事件 (也让窗口有机会刷新) <<<<< ---
        # 等待1毫秒，如果用户按下 'q' 键，则退出循环
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("用户按下 'q'，退出。")
            break

    except Exception as e:
        print(f"处理帧时发生错误: {e}")
        break # 如果处理出错，也退出循环


# --- 循环结束后，释放资源 ---
cap.release()             # 关闭视频文件
cv2.destroyAllWindows()   # 关闭所有 OpenCV 创建的窗口


# --- 打印最终计数 ---
# 如果选择了水平线
if line_orientation == 'h':
    print(f"最终计数: 上到下 = {up_to_down_count}, 下到上 = {down_to_up_count}")
# 如果选择了垂直线
if line_orientation == 'v':
    print(f"最终计数: 左到右 = {left_to_right_count}, 右到左 = {right_to_left_count}")