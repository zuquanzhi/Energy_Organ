import os
import cv2
import argparse
import numpy as np
import glob
from pathlib import Path


# ===== 绘制不同格式的标签 =====
def draw_aabb_label(image, label_line, img_w, img_h):
    """绘制YOLO标准矩形框 (class_id + x_center y_center width height)"""
    parts = label_line.strip().split()
    if len(parts) != 5:
        print(f"AABB标签格式错误: {label_line}")
        return

    try:
        class_id = int(float(parts[0]))
        xc = float(parts[1]) * img_w
        yc = float(parts[2]) * img_h
        w = float(parts[3]) * img_w
        h = float(parts[4]) * img_h

        x1 = int(xc - w / 2)
        y1 = int(yc - h / 2)
        x2 = int(xc + w / 2)
        y2 = int(yc + h / 2)

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 2)  # 黄色框
        cv2.putText(image, f"{class_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    except Exception as e:
        print(f"绘制AABB时出错: {e}")

def draw_point_label(image, label_line, img_w, img_h):
    """绘制单点格式 (class_id x y) 或 多点格式"""
    parts = label_line.strip().split()
    
    # 检查是否至少有类别ID和一组坐标
    if len(parts) < 3:
        print(f"Point标签格式错误，数据不足: {label_line}")
        return

    try:
        # 解析类别ID
        class_id = int(float(parts[0]))
        
        # 检查是否是标准单点格式 (class_id x y)
        if len(parts) == 3:
            x = float(parts[1]) * img_w
            y = float(parts[2]) * img_h
            
            # 绘制单个点
            cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1)  # 绿色点
            cv2.putText(image, f"ID: {class_id}", (int(x)+5, int(y)-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            # 如果有更多坐标，可能是多点格式或其他格式
            # 尝试将其作为一系列点绘制出来
            points = []
            for i in range(1, len(parts), 2):
                if i+1 < len(parts):
                    try:
                        x = float(parts[i]) * img_w
                        y = float(parts[i+1]) * img_h
                        points.append((int(x), int(y)))
                    except ValueError:
                        # 跳过无法解析为浮点数的值
                        pass
            
            # 绘制多个点
            colors = [
                (0, 255, 0),    # 绿色
                (255, 0, 0),    # 蓝色
                (0, 0, 255),    # 红色
                (255, 255, 0),  # 青色
                (0, 255, 255),  # 黄色
                (255, 0, 255)   # 洋红色
            ]
            
            for i, point in enumerate(points):
                color_idx = i % len(colors)
                cv2.circle(image, point, 4, colors[color_idx], -1)
                cv2.putText(image, f"{i}", (point[0]+5, point[1]-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[color_idx], 1)
            
            # 如果是第一个点，显示类别ID
            if points:
                cv2.putText(image, f"ID: {class_id}", (points[0][0], points[0][1]-20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    except Exception as e:
        print(f"绘制Point格式时出错: {e}")


def draw_pose_label(image, label_line, img_w, img_h):
    """绘制Pose格式
    两种格式:
    1. Dim=2: class_id x y w h px1 py1 px2 py2 ...
    2. Dim=3: class_id x y w h px1 py1 v1 px2 py2 v2 ...
    """
    parts = label_line.strip().split()
    if len(parts) < 5:  # 至少需要类别ID和边界框信息
        print(f"Pose标签格式错误，数据不足: {label_line}")
        return

    try:
        # 解析类别ID和边界框
        class_id = int(float(parts[0]))
        box_x = float(parts[1]) * img_w
        box_y = float(parts[2]) * img_h
        box_w = float(parts[3]) * img_w
        box_h = float(parts[4]) * img_h

        # 绘制边界框
        x1 = int(box_x - box_w / 2)
        y1 = int(box_y - box_h / 2)
        x2 = int(box_x + box_w / 2)
        y2 = int(box_y + box_h / 2)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 165, 255), 1)  # 橙色框，细线

        # 解析关键点
        keypoints = []
        visibilities = []
        
        # 检查是否为 Dim=3 格式（带可见性）
        is_dim3 = (len(parts) - 5) % 3 == 0 and len(parts) >= 8
        
        if is_dim3:  # Dim=3 格式: class_id x y w h px1 py1 v1 px2 py2 v2 ...
            for i in range(5, len(parts), 3):
                if i + 2 >= len(parts):
                    break
                x = float(parts[i]) * img_w
                y = float(parts[i + 1]) * img_h
                v = int(float(parts[i + 2]))  # 可见性标志
                keypoints.append((x, y))
                visibilities.append(v)
        else:  # Dim=2 格式: class_id x y w h px1 py1 px2 py2 ...
            for i in range(5, len(parts), 2):
                if i + 1 >= len(parts):
                    break
                x = float(parts[i]) * img_w
                y = float(parts[i + 1]) * img_h
                keypoints.append((x, y))
                visibilities.append(2)  # 默认可见

        # 绘制关键点和连线
        colors = [
            (0, 0, 255),    # 红色
            (0, 255, 0),    # 绿色
            (255, 0, 0),    # 蓝色
            (255, 255, 0),  # 青色
            (255, 0, 255),  # 洋红
            (0, 255, 255),  # 黄色
            (255, 165, 0),  # 橙色
            (128, 0, 128)   # 紫色
        ]

        # 绘制关键点
        for i, ((x, y), v) in enumerate(zip(keypoints, visibilities)):
            if v > 0:  # 如果点可见
                color_idx = i % len(colors)
                cv2.circle(image, (int(x), int(y)), 4, colors[color_idx], -1)
                cv2.putText(image, str(i), (int(x) + 5, int(y) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[color_idx], 1)

        # 如果有多个关键点，可以考虑绘制连线表示骨架
        # 这需要根据具体应用场景定义连接关系
        # 例如: 如果是人体骨架，可以定义特定的连线

        # 显示类别标签
        cv2.putText(image, f"ID: {class_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                    
    except Exception as e:
        print(f"绘制Pose时出错: {e}")
        import traceback
        traceback.print_exc()


def draw_obb_label(image, label_line, img_w, img_h):
    """绘制YOLO OBB格式 (class_id + 8个归一化坐标)"""
    parts = label_line.strip().split()
    if len(parts) != 9:
        print(f"OBB标签格式错误: {label_line}")
        return

    try:
        class_id = int(float(parts[0]))
        coords_norm = [float(p) for p in parts[1:]]

        box_points_abs = []
        for i in range(0, 8, 2):
            x_abs = coords_norm[i] * img_w
            y_abs = coords_norm[i+1] * img_h
            box_points_abs.append([x_abs, y_abs])

        box_np = np.array(box_points_abs, dtype=np.int32)
        cv2.drawContours(image, [box_np], 0, (255, 0, 255), 2)

        cv2.putText(image, f"ID: {class_id}", (box_np[0][0], box_np[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
    except Exception as e:
        print(f"绘制OBB时出错: {e}")


# ===== 主要可视化逻辑 =====

def visualize_labels(img_path, label_path, output_path=None, show=True, label_format="obb"):
    image = cv2.imread(img_path)
    if image is None:
        print(f"无法加载图像: {img_path}")
        return False

    img_h, img_w = image.shape[:2]
    vis_image = image.copy()

    if not os.path.exists(label_path):
        print(f"标签文件不存在: {label_path}")
        return False

    with open(label_path, 'r') as f:
        lines = f.readlines()

    draw_func_map = {
    "point": draw_point_label,
    "pose": draw_pose_label,
    "obb": draw_obb_label,
    "aabb": draw_aabb_label
    }
    draw_func = draw_func_map.get(label_format, draw_obb_label)

    for line in lines:
        line = line.strip()
        if not line:
            continue
        draw_func(vis_image, line, img_w, img_h)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, vis_image)
        print(f"已保存可视化结果到: {output_path}")

    if show:
        global current_idx, image_files
        while True:
            cv2.imshow("Label Viewer", vis_image)
            key = cv2.waitKey(0)
            if key == 27:  # ESC
                return False
            elif key == 81 or key == ord('a'):  # Left
                current_idx -= 1
                break
            elif key == 83 or key == ord('d'):  # Right
                current_idx += 1
                break
            else:
                continue
    return True


# ===== 批量处理 =====

image_files = []
current_idx = 0

def batch_visualize(image_dir, label_dir, output_dir=None, show=True, label_format="obb"):
    global image_files, current_idx
    image_exts = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
    image_files = []

    for ext in image_exts:
        image_files.extend(glob.glob(os.path.join(image_dir, ext)))
        image_files.extend(glob.glob(os.path.join(image_dir, '**', ext), recursive=True))

    image_files.sort()
    total = len(image_files)
    if total == 0:
        print(f"找不到图像文件: {image_dir}")
        return

    print(f"找到 {total} 张图像")

    while 0 <= current_idx < total:
        img_path = image_files[current_idx]
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(label_dir, f"{base_name}.txt")
        output_path = os.path.join(output_dir, f"{base_name}_vis.jpg") if output_dir else None

        print(f"[{current_idx+1}/{total}] 正在显示: {img_path}")
        success = visualize_labels(img_path, label_path, output_path, show=show, label_format=label_format)
        if not success:
            break


# ===== 入口函数 =====

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="可视化YOLO格式标签（point/pose/obb/aabb）")
    parser.add_argument("--image", type=str, required=True, help="输入图像路径或目录")
    parser.add_argument("--label", type=str, required=True, help="输入标签路径或目录")
    parser.add_argument("--output", type=str, help="输出可视化结果目录")
    parser.add_argument("--format", choices=["point", "pose", "obb", "aabb"], default="aabb", help="标签格式")
    parser.add_argument("--no-show", action="store_true", help="不显示图像窗口")

    args = parser.parse_args()

    if os.path.isdir(args.image) and os.path.isdir(args.label):
        batch_visualize(args.image, args.label, args.output, show=not args.no_show, label_format=args.format)
    elif os.path.isfile(args.image) and os.path.isfile(args.label):
        visualize_labels(args.image, args.label, args.output, show=not args.no_show, label_format=args.format)
    else:
        print("错误：请确保图像和标签路径均为文件或目录，并保持一致类型。")