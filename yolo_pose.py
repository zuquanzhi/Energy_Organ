import os
import json
import argparse
import glob
import numpy as np
import cv2
import shutil
import yaml
from pathlib import Path

def load_config(config_path):
    """加载 YAML 配置文件"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        if not config:
            raise ValueError("配置文件为空或格式错误")
        print("配置加载成功。")
        return config
    except Exception as e:
        raise ValueError(f"加载配置文件时出错: {e}")

def build_mappings(config):
    """从配置构建类别和索引映射"""
    if "class_definitions" not in config:
        raise ValueError("配置文件缺少 'class_definitions'")

    annotation_id_to_class_id = {}
    annotation_id_to_indices = {}
    class_id_to_name = {}
    class_names = []

    current_class_id = 0
    for ann_id, definition in config["class_definitions"].items():
        if "class_name" not in definition or "indices" not in definition:
            print(f"警告: 跳过无效的类别定义 (annotation_id: {ann_id})，缺少 'class_name' 或 'indices'")
            continue
        print(f"处理类别定义 (annotation_id: {ann_id})",f"{definition}")
        class_name = definition["class_name"]
        indices = definition["indices"]

        if not isinstance(indices, list) or not all(isinstance(i, int) for i in indices):
            #  print(f"警告: 跳过无效的类别定义 (annotation_id: {ann_id})，'indices' 必须是整数列表")
             continue
        if len(indices) < 3:
            #  print(f"警告: 类别定义 (annotation_id: {ann_id}) 的索引数量少于3个，可能无法确定有效矩形")
            continue

        # 分配 Class ID (如果名称已存在，则复用)
        if class_name not in class_names:
            class_names.append(class_name)
            class_id = current_class_id
            class_id_to_name[class_id] = class_name
            current_class_id += 1
        else:
            class_id = class_names.index(class_name)

        annotation_id_to_class_id[ann_id] = class_id
        annotation_id_to_indices[ann_id] = indices
        print(f"映射: 标注ID '{ann_id}' -> 类别ID {class_id} ('{class_name}'), 使用索引: {indices}")

    if not class_names:
         raise ValueError("配置文件中没有定义有效的类别。")

    return annotation_id_to_class_id, annotation_id_to_indices, class_names

def extract_data_from_json(json_path, annotation_id_to_class_id, annotation_id_to_indices):
    """从JSON文件中提取图像信息和基于指定索引的关键点"""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"无法加载或解析JSON文件 {json_path}: {e}")
        return None

    result = {
        "width": 0,
        "height": 0,
        "image_path": "",
        "objects": []
    }

    # 查找相机捕获
    capture = None
    for cap in data.get('captures', []):
        if cap.get('@type') == "type.unity.com/unity.solo.RGBCamera":
            capture = cap
            break
    
    if not capture:
        print(f"警告: 在 {json_path} 中未找到 RGBCamera 捕获。")
        return None

    # 获取图像尺寸和路径
    if "dimension" in capture:
        result["width"] = int(capture["dimension"][0])
        result["height"] = int(capture["dimension"][1])
    if "filename" in capture:
        base_dir = os.path.dirname(json_path)
        result["image_path"] = os.path.join(base_dir, capture["filename"])
        # 检查图像文件是否存在，如果不存在则尝试查找
        if not os.path.exists(result["image_path"]):
             print(f"警告: 图像文件 {result['image_path']} 不存在，将尝试查找...")
             found = False
             for ext in ['.png', '.jpg', '.jpeg']:
                  potential_path = Path(json_path).with_suffix(ext)
                  if potential_path.exists():
                       result["image_path"] = str(potential_path)
                       print(f"找到图像文件: {result['image_path']}")
                       # 如果尺寸未知，尝试读取
                       if result["width"] == 0 or result["height"] == 0:
                            try:
                                img = cv2.imread(result["image_path"])
                                if img is not None:
                                     result["height"], result["width"] = img.shape[:2]
                            except Exception as img_e:
                                 print(f"尝试读取图像尺寸时出错: {img_e}")
                       found = True
                       break
             if not found:
                  print(f"错误: 无法找到 {os.path.basename(json_path)} 对应的图像文件。")
                  return None

    if result["width"] <= 0 or result["height"] <= 0:
         print(f"错误: {json_path} 的图像尺寸无效 ({result['width']}x{result['height']})。")
         return None

    # 处理标注
    for annotation in capture.get("annotations", []):
        annotation_id = annotation.get("id")
        
        # 检查此标注ID是否在我们的配置中定义
        if annotation_id in annotation_id_to_class_id:
            class_id = annotation_id_to_class_id[annotation_id]
            required_indices = annotation_id_to_indices[annotation_id]
            
            for instance in annotation.get("values", []):
                if not instance: continue
                
                instance_id = instance.get("instanceId", "N/A")
                keypoints_map = {kp['index']: kp for kp in instance.get("keypoints", []) if 'index' in kp and 'location' in kp and kp.get('state', 0) > 0}

                # 提取所需索引的关键点坐标
                selected_points = []
                found_indices = []
                for index in required_indices:
                    if index in keypoints_map:
                        kp = keypoints_map[index]
                        loc = kp['location']
                        if isinstance(loc, (list, tuple)) and len(loc) == 2:
                            try:
                                selected_points.append((float(loc[0]), float(loc[1])))
                                found_indices.append(index)
                            except (ValueError, TypeError):
                                # print(f"警告: 实例 {instance_id}, 标注ID '{annotation_id}', 索引 {index} 的坐标无效: {loc}")
                                continue
                        else:
                            continue
                             
                            #  print(f"警告: 实例 {instance_id}, 标注ID '{annotation_id}', 索引 {index} 的坐标格式无效: {loc}")
                    else:
                        continue
                        # print(f"警告: 实例 {instance_id}, 标注ID '{annotation_id}', 未找到或状态无效的索引: {index}")

                # 检查是否找到了足够数量的点
                if len(selected_points) >= 3:  # 至少需要3个点来定义一个矩形
                    result["objects"].append({
                        "class_id": class_id,
                        "points": selected_points,
                        "annotation_id": annotation_id,
                        "instance_id": instance_id
                    })
                elif len(selected_points) > 0:
                     print(f"警告: 实例 {instance_id}, 标注ID '{annotation_id}', 找到的点数 ({len(selected_points)}) 不足以形成矩形 (需要至少3个)。")

    return result

def convert_to_yolo_pose(data):
    """
    将提取的数据转换为YOLO格式
    返回：YOLO 标签格式字符串
    """
    if not data or "objects" not in data or not data["objects"]:
        return []

    img_width = data["width"]
    img_height = data["height"]

    if img_width <= 0 or img_height <= 0:
        print("错误: 图像尺寸无效，无法进行归一化。")
        return []

    yolo_poses = []
    
    # 按照class_id分组对象
    grouped_objects = {}
    for obj in data["objects"]:
        class_id = obj["class_id"]
        if class_id not in grouped_objects:
            grouped_objects[class_id] = []
        grouped_objects[class_id].append(obj)
    
    # 处理每个类别的所有对象
    for class_id, objects in grouped_objects.items():
        for obj in objects:
            points = obj["points"]
            if not points:
                print(f"警告: 实例 {obj.get('instance_id', 'N/A')} 没有关键点，跳过。")
                continue
            
            # 计算边界框
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            # 计算中心点、宽度和高度
            box_center_x = (x_min + x_max) / 2
            box_center_y = (y_min + y_max) / 2
            box_width = max(1.0, x_max - x_min)  # 确保宽度至少为1
            box_height = max(1.0, y_max - y_min)  # 确保高度至少为1
            
            # 归一化边界框坐标
            box_center_x_norm = box_center_x / img_width
            box_center_y_norm = box_center_y / img_height
            box_width_norm = box_width / img_width
            box_height_norm = box_height / img_height
            
            # 边界检查边界框
            box_center_x_norm = np.clip(box_center_x_norm, 0.0, 1.0)
            box_center_y_norm = np.clip(box_center_y_norm, 0.0, 1.0)
            box_width_norm = np.clip(box_width_norm, 0.0, 1.0)
            box_height_norm = np.clip(box_height_norm, 0.0, 1.0)
            
            # 创建标签开头（类别ID + 边界框）
            pose_values = [
                f"{class_id}",
                f"{box_center_x_norm:.6f}",
                f"{box_center_y_norm:.6f}",
                f"{box_width_norm:.6f}",
                f"{box_height_norm:.6f}"
            ]
            
            # 将所有关键点归一化并添加到标签中
            for x, y in points:
                x_norm = x / img_width
                y_norm = y / img_height
                
                # 边界检查关键点坐标
                x_norm = np.clip(x_norm, 0.0, 1.0)
                y_norm = np.clip(y_norm, 0.0, 1.0)
                
                pose_values.append(f"{x_norm:.6f}")
                pose_values.append(f"{y_norm:.6f}")
            
            # 创建YOLO Pose格式字符串 - 每个对象生成一行
            yolo_pose_line = " ".join(pose_values)
            yolo_poses.append(yolo_pose_line)

    return yolo_poses

def process_files(config):
    """处理配置文件中指定的JSON文件或目录"""
    input_path = config.get("input", "")
    output_dir = config.get("output", "output_yolo")
    copy_images = config.get("copy_images", True)

    if not input_path:
        print("错误: 配置文件中未指定 'input' 路径。")
        return

    # 构建映射关系
    try:
        annotation_id_to_class_id, annotation_id_to_indices, class_names = build_mappings(config)
    except ValueError as e:
        print(f"错误: {e}")
        return

    # 查找JSON文件
    json_files = []
    if os.path.isdir(input_path):
        print(f"正在搜索目录 {input_path} 中的 .json 文件...")
        json_files = glob.glob(os.path.join(input_path, '**', '*.json'), recursive=True)
    elif os.path.isfile(input_path) and input_path.endswith('.json'):
        json_files = [input_path]
    
    if not json_files:
        print(f"在路径 {input_path} 中未找到 JSON 文件。")
        return
    
    print(f"找到 {len(json_files)} 个 JSON 文件进行处理。")

    # 创建输出目录
    labels_dir = os.path.join(output_dir, "labels")
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(labels_dir, exist_ok=True)
    if copy_images:
        os.makedirs(images_dir, exist_ok=True)

    processed_count = 0
    skipped_count = 0
    total_objects = 0

    # 处理每个JSON文件
    for json_path in sorted(json_files):
        # print(f"\n处理文件: {json_path}")
        
        # 提取数据
        extracted_data = extract_data_from_json(json_path, annotation_id_to_class_id, annotation_id_to_indices)
        
        if not extracted_data or not extracted_data["objects"]:
            print(f"文件 {json_path} 中未提取到有效对象或数据，跳过。")
            skipped_count += 1
            continue

        # 转换为YOLO格式
        yolo_obb_lines = convert_to_yolo_pose(extracted_data)

        if not yolo_obb_lines:
            print(f"文件 {json_path} 未生成有效的 YOLO 标签，跳过。")
            skipped_count += 1
            continue

        # 保存标签文件
        base_name = Path(json_path).stem.replace('.frame_data', '')  # 移除常见后缀
        label_path = os.path.join(labels_dir, f"{base_name}.txt")
        try:
            with open(label_path, 'w') as f:
                f.write('\n'.join(yolo_obb_lines))
            # print(f"已保存标签到: {label_path}")
            processed_count += 1
            total_objects += len(yolo_obb_lines)
        except Exception as e:
            print(f"保存标签文件 {label_path} 时出错: {e}")
            skipped_count += 1
            continue

        # 复制图像文件
        if copy_images and extracted_data["image_path"] and os.path.exists(extracted_data["image_path"]):
            src_img_path = extracted_data["image_path"]
            img_filename = os.path.basename(src_img_path)
            # 确保目标文件名与标签文件名（不含扩展名）匹配
            dst_img_filename = f"{base_name}{Path(img_filename).suffix}"
            dst_img_path = os.path.join(images_dir, dst_img_filename)
            try:
                shutil.copy2(src_img_path, dst_img_path)
            except Exception as e:
                print(f"复制图像 {src_img_path} 到 {dst_img_path} 时出错: {e}")
        elif copy_images:
            print(f"警告: 未找到或无法访问图像文件 {extracted_data.get('image_path', 'N/A')}，无法复制。")

    # 创建 data.yaml 文件
    yaml_path = os.path.join(output_dir, "data.yaml")
    try:
        with open(yaml_path, 'w') as f:
            f.write(f"train: ./images\n")
            f.write(f"val: ./images\n")
            f.write(f"test: # 可选\n\n")
            f.write(f"nc: {len(class_names)}\n")
            f.write(f"names:\n")
            for i, name in enumerate(class_names):
                f.write(f"  {i}: {name}\n")  # 使用 ID: name 格式更标准
        print(f"\n已生成 data.yaml 文件: {yaml_path}")
    except Exception as e:
        print(f"生成 data.yaml 文件时出错: {e}")

    print("\n处理完成。")
    print(f"成功处理文件数: {processed_count}")
    print(f"跳过文件数: {skipped_count}")
    print(f"总共生成对象数: {total_objects}")
    print(f"输出目录: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="根据配置文件中指定的关键点索引将 Unity Perception JSON 转换为 YOLO 格式。")
    parser.add_argument("config", help="指向 YAML 配置文件的路径。")
    
    args = parser.parse_args()
    
    try:
        config = load_config(args.config)
        process_files(config)
    except (FileNotFoundError, ValueError, yaml.YAMLError) as e:
        print(f"错误: {e}")
    except Exception as e:
        print(f"发生意外错误: {e}")

if __name__ == "__main__":
    main()  # 输入 JSON 文件或包含 JSON 文件的目录
