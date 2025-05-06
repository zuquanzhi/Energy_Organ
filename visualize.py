# visualize.py - 可视化YOLO OBB格式标签

import os
import cv2
import argparse
import numpy as np
import glob
from pathlib import Path


def draw_obb_label(image, label_line, img_w, img_h):
    """绘制YOLO OBB（定向边界框）标签，格式为 class_id x1 y1 x2 y2 x3 y3 x4 y4"""
    parts = label_line.strip().split()
    if len(parts) != 9: # 1 class_id + 8 coordinates
        print(f"OBB标签格式错误 (期望9个部分): {label_line}")
        return
    
    try:
        # 解析OBB参数
        class_id = int(float(parts[0]))
        
        # 解析归一化的8个坐标值
        coords_norm = [float(p) for p in parts[1:]]
        
        # 将坐标反归一化并重塑为点数组
        # (x1, y1, x2, y2, x3, y3, x4, y4)
        box_points_abs = []
        for i in range(0, 8, 2):
            x_abs = coords_norm[i] * img_w
            y_abs = coords_norm[i+1] * img_h
            box_points_abs.append([x_abs, y_abs])
        
        # 转换为Numpy数组，并确保数据类型为int32以便绘制
        box_np = np.array(box_points_abs, dtype=np.int32)
        
        # 绘制OBB (多边形)
        cv2.drawContours(image, [box_np], 0, (255, 0, 255), 2) # 洋红色
        
        # 计算中心点用于放置标签文本 (可选，也可以使用第一个点或左上角点)
        center_x = int(np.mean(box_np[:, 0]))
        center_y = int(np.mean(box_np[:, 1]))
        
        # 绘制一个小的中心点（可选）
        cv2.circle(image, (center_x, center_y), 3, (0, 0, 255), -1) # 红色小点
        
        # 添加类别标签
        label_text = f"ID: {class_id}"
        cv2.putText(image, label_text, (box_np[0][0], box_np[0][1] - 10), # 在第一个点上方显示
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
    except ValueError as ve:
        print(f"解析标签时值错误: {ve} in line: {label_line}")
    except Exception as e:
        print(f"绘制OBB时出错: {e} in line: {label_line}")


def visualize_labels(img_path, label_path, output_path=None, show=True):
    """可视化OBB标签图像"""
    try:
        # 加载图像
        image = cv2.imread(img_path)
        if image is None:
            print(f"无法加载图像: {img_path}")
            return False
        
        img_h, img_w = image.shape[:2]
        
        # 检查标签文件
        if not os.path.exists(label_path):
            print(f"标签文件不存在: {label_path}")
            return False
        
        # 读取标签文件
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"读取标签文件时出错: {e}")
            return False
            
        # 创建副本用于绘制
        vis_image = image.copy()
        
        # 处理每行标签
        for line in lines:
            line = line.strip()
            if not line:
                continue
            draw_obb_label(vis_image, line, img_w, img_h)
        
        # 添加标题信息
        cv2.putText(vis_image, "YOLO OBB Detection", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 保存结果
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, vis_image)
            print(f"已保存可视化结果到: {output_path}")
        
        # 显示结果
        if show:
            cv2.imshow(f"YOLO OBB - {os.path.basename(img_path)}", vis_image)
            print("按Esc键退出，或按任意键继续...")
            key = cv2.waitKey(0)
            if key == 27:  # Esc键
                cv2.destroyAllWindows()
                return False  # 表示用户选择退出
        
        return True
    except Exception as e:
        print(f"可视化过程中出错: {e}")
        return False


def batch_visualize(image_dir, label_dir, output_dir=None, show=True):
    """批量可视化OBB标签"""
    # 查找所有图像文件
    image_exts = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
    image_files = []
    
    for ext in image_exts:
        image_files.extend(glob.glob(os.path.join(image_dir, ext)))
        image_files.extend(glob.glob(os.path.join(image_dir, '**', ext), recursive=True))
    
    if not image_files:
        print(f"在目录 {image_dir} 中找不到图像文件")
        return
    
    image_files = sorted(image_files)
    print(f"找到 {len(image_files)} 个图像文件")
    
    # 处理每个图像
    for img_path in image_files:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(label_dir, f"{base_name}.txt")
        
        if not os.path.exists(label_path):
            print(f"跳过 {img_path}，找不到对应的标签文件")
            continue
        
        # 确定输出路径
        output_path = None
        if output_dir:
            output_path = os.path.join(output_dir, f"{base_name}_obb.jpg")
        
        # 可视化
        print(f"可视化: {img_path} 与 {label_path}")
        if not visualize_labels(img_path, label_path, output_path, show):
            break


def visualize_directory(img_dir, label_dir, output_dir=None, show=True):
    """可视化指定目录中的所有图像和标签"""
    if os.path.isdir(img_dir) and os.path.isdir(label_dir):
        batch_visualize(img_dir, label_dir, output_dir, show)
    else:
        print(f"图像目录 {img_dir} 或标签目录 {label_dir} 不存在")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="可视化 YOLO OBB 格式的标签")
    parser.add_argument("--image", type=str, help="输入图像路径或图像目录")
    parser.add_argument("--label", type=str, help="输入标签文件路径或标签目录")
    parser.add_argument("--output", type=str, help="输出可视化结果的目录或文件路径")
    parser.add_argument("--no-show", action="store_true", help="不显示可视化结果窗口")

    args = parser.parse_args()
    
    # 检查输入参数
    if not args.image or not args.label:
        parser.print_help()
        exit(1)
    
    # 检查是目录还是单个文件
    if os.path.isdir(args.image) and os.path.isdir(args.label):
        # 批量处理目录
        visualize_directory(args.image, args.label, args.output, not args.no_show)
    elif os.path.isfile(args.image) and os.path.isfile(args.label):
        # 处理单个文件
        visualize_labels(args.image, args.label, args.output, not args.no_show)
    else:
        print("错误: 请同时指定图像和标签文件，或同时指定图像和标签目录")