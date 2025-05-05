# visualize.py - 可视化YOLO OBB格式标签

import os
import cv2
import argparse
import numpy as np
import glob
from pathlib import Path


def draw_obb_label(image, label_line, img_w, img_h):
    """绘制YOLO OBB（定向边界框）标签"""
    parts = label_line.strip().split()
    if len(parts) < 6:
        print(f"OBB标签格式错误: {label_line}")
        return
    
    try:
        # 解析OBB参数
        class_id = int(float(parts[0]))
        xc = float(parts[1]) * img_w
        yc = float(parts[2]) * img_h
        w = float(parts[3]) * img_w
        h = float(parts[4]) * img_h
        angle = float(parts[5])  # 弧度
        
        # 转换角度到OpenCV格式（度）
        angle_deg = np.rad2deg(angle)
        
        # 构造旋转矩形
        rect = ((xc, yc), (w, h), angle_deg)
        
        # 获取四个角点
        box = cv2.boxPoints(rect)
        # 使用np.int32代替np.int0
        box = np.array(box, dtype=np.int32)
        
        # 绘制OBB
        cv2.drawContours(image, [box], 0, (255, 0, 255), 2)
        
        # 绘制中心点
        cv2.circle(image, (int(xc), int(yc)), 5, (0, 0, 255), -1)
        
        # 添加类别标签
        label_text = f"Class: {class_id}"
        cv2.putText(image, label_text, (int(xc), int(yc) - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
    except Exception as e:
        print(f"绘制OBB时出错: {e}")


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