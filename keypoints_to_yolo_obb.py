import os
import json
import cv2
import numpy as np
import argparse
from pathlib import Path
from shutil import copy2
import yaml

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def calculate_obb_from_points(points):
    """使用cv2.minAreaRect从一组点计算OBB的四个角点"""
    if not points or len(points) < 3:
        return None

    points_array = np.array(points, dtype=np.float32)
    
    try:
        # 计算最小面积旋转矩形
        # 返回: ((center_x, center_y), (width, height), angle_degrees)
        rect = cv2.minAreaRect(points_array)
        
        # 获取四个角点
        # box_points 是一个 4x2 的数组 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        box_points = cv2.boxPoints(rect)
        return box_points

    except Exception as e:
        print(f"计算 minAreaRect 或 boxPoints 时出错: {e}")
        return None

def convert_to_yolo_obb(data, class_definitions_with_numeric_id):
    """将提取的数据转换为YOLO OBB格式 (numeric_class_id x1 y1 x2 y2 x3 y3 x4 y4)"""
    if not data or "objects" not in data or not data["objects"]:
        return []

    img_width = data["width"]
    img_height = data["height"]

    if img_width <= 0 or img_height <= 0:
        print("错误: 图像尺寸无效，无法进行归一化。")
        return []

    yolo_obbs = []
    for obj in data["objects"]:
        annotation_id = obj["annotation_id"]
        if annotation_id not in class_definitions_with_numeric_id:
            print(f"警告: 未找到 annotation_id 为 {annotation_id} 的类别定义。")
            continue

        class_info = class_definitions_with_numeric_id[annotation_id]
        numeric_class_id = class_info["class_id"] # 使用数字类别ID
        indices = class_info["indices"]

        # 提取指定索引的关键点
        # 确保 obj["keypoints"] 是一个列表，并且索引有效
        points = []
        if isinstance(obj.get("keypoints"), list):
            for i in indices:
                if 0 <= i < len(obj["keypoints"]):
                    # 确保 keypoint 本身是有效的坐标对
                    kp_loc = obj["keypoints"][i]
                    if isinstance(kp_loc, (list, tuple)) and len(kp_loc) == 2:
                        try:
                            points.append((float(kp_loc[0]), float(kp_loc[1])))
                        except (ValueError, TypeError):
                             print(f"警告: 实例 {obj.get('instance_id', 'N/A')}, 标注ID '{annotation_id}', 索引 {i} 的关键点坐标无效: {kp_loc}")
                    else:
                        print(f"警告: 实例 {obj.get('instance_id', 'N/A')}, 标注ID '{annotation_id}', 索引 {i} 的关键点格式无效: {kp_loc}")

                else:
                    print(f"警告: 实例 {obj.get('instance_id', 'N/A')}, 标注ID '{annotation_id}', 索引 {i} 超出范围 (关键点数量: {len(obj['keypoints'])})")
        else:
            print(f"警告: 实例 {obj.get('instance_id', 'N/A')}, 标注ID '{annotation_id}', keypoints 数据缺失或格式不正确。")


        if len(points) < 3: # 需要至少3个点来形成矩形
            print(f"警告: 实例 {obj.get('instance_id', 'N/A')}, 标注ID '{annotation_id}', 提取到的有效点数 ({len(points)}) 不足以形成矩形。所需索引: {indices}")
            continue
            
        # 从对象中定义的点计算OBB的四个角点
        box_coords = calculate_obb_from_points(points)
        
        if box_coords is not None and len(box_coords) == 4:
            normalized_coords = []
            # valid_box = True # 移除了未使用变量
            for point_coord in box_coords: # 重命名 'point' 避免与外层 'points' 冲突
                x_norm = point_coord[0] / img_width
                y_norm = point_coord[1] / img_height
                
                # 裁剪到 [0, 1] 范围
                x_norm = np.clip(x_norm, 0.0, 1.0)
                y_norm = np.clip(y_norm, 0.0, 1.0)
                normalized_coords.extend([x_norm, y_norm])
            
            coord_str = " ".join([f"{c:.6f}" for c in normalized_coords])
            yolo_obb_line = f"{numeric_class_id} {coord_str}" # 使用数字类别ID
            yolo_obbs.append(yolo_obb_line)
        else:
            print(f"警告: 无法为实例 {obj.get('instance_id', 'N/A')} (标注ID: {obj.get('annotation_id')}) 计算有效的4个角点。")

    return yolo_obbs

def process_json_file(json_path, output_dir, class_definitions_with_numeric_id, copy_images):
    """处理单个 JSON 文件并生成 YOLO OBB 标签"""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"错误: 无法加载或解析JSON文件 {json_path}: {e}")
        return False


    # 提取图像文件名和尺寸
    image_filename = None
    img_width = 0
    img_height = 0
    image_path = None

    for capture in data.get('captures', []):
        if capture.get('@type') == "type.unity.com/unity.solo.RGBCamera":
            image_filename = capture.get('filename')
            if "dimension" in capture:
                try:
                    img_width = int(capture["dimension"][0])
                    img_height = int(capture["dimension"][1])
                except (ValueError, TypeError):
                    print(f"警告: JSON文件 {json_path} 中的图像尺寸无效。")
            break
    
    if not image_filename:
        print(f"错误: 在 JSON 文件 {json_path} 中找不到RGB相机捕获或图像文件名。")
        return False

    json_dir = os.path.dirname(json_path)
    image_path = os.path.join(json_dir, image_filename)

    if not os.path.exists(image_path) or (img_width <= 0 or img_height <= 0):
        print(f"警告: 图像文件 {image_path} 不存在或JSON中尺寸无效。尝试从图像读取尺寸...")
        try:
            img = cv2.imread(image_path)
            if img is not None:
                img_height, img_width = img.shape[:2]
                print(f"从图像文件成功读取尺寸: {img_width}x{img_height}")
            else:
                print(f"错误: 无法加载图像文件 {image_path} 以获取尺寸。")
                return False
        except Exception as e:
            print(f"错误: 读取图像文件 {image_path} 以获取尺寸时出错: {e}")
            return False

    if img_width <= 0 or img_height <= 0:
        print(f"错误: 无法确定图像 {image_filename} 的有效尺寸。")
        return False

    # 提取关键点数据
    objects_for_yolo = []
    for capture in data.get('captures', []): # 再次迭代以获取标注
        for annotation in capture.get('annotations', []):
            if annotation.get('@type') == "type.unity.com/unity.solo.KeypointAnnotation":
                annotation_id_str = annotation.get('id')
                if annotation_id_str not in class_definitions_with_numeric_id:
                    continue # 跳过未在配置中定义的标注ID

                for instance in annotation.get('values', []):
                    keypoints_loc_list = [] # 存储 (x,y) 坐标
                    # 预填充一个足够大的列表，用None标记缺失的关键点，以保持索引对应
                    # 假设最大索引不会太大，或者从数据中动态确定最大索引
                    max_idx_seen = 0
                    temp_kps = {}

                    for keypoint in instance.get('keypoints', []):
                        state = keypoint.get('state', 0)
                        location = keypoint.get('location')
                        idx = keypoint.get('index', -1)

                        if idx != -1:
                             max_idx_seen = max(max_idx_seen, idx)
                        if state in [1, 2] and location and len(location) == 2:
                            try:
                                temp_kps[idx] = (float(location[0]), float(location[1]))
                            except (ValueError, TypeError):
                                pass # 忽略无效坐标
                    
                    # 创建最终的关键点列表，保持索引
                    # 如果需要处理非常稀疏的大索引，此方法可能需要优化
                    if temp_kps: # 如果至少有一个有效关键点
                        keypoints_loc_list = [temp_kps.get(i) for i in range(max_idx_seen + 1)]


                    objects_for_yolo.append({
                        "annotation_id": annotation_id_str,
                        "instance_id": instance.get('instance_id'),
                        "keypoints": keypoints_loc_list # 这是 (x,y) 坐标的列表，或None
                    })
    
    if not objects_for_yolo:
        print(f"文件 {json_path} 中未找到与配置匹配的有效对象。")
        return True # 算作已处理但无输出

    # 转换为 YOLO OBB 格式
    yolo_data_for_conversion = {
        "width": img_width,
        "height": img_height,
        "objects": objects_for_yolo
    }
    yolo_obbs = convert_to_yolo_obb(yolo_data_for_conversion, class_definitions_with_numeric_id)

    if not yolo_obbs:
        print(f"文件 {json_path} 未生成YOLO OBB标签。")
        return True # 算作已处理但无输出

    # 保存 YOLO OBB 标签文件
    base_filename = Path(json_path).stem.replace('.frame_data', '')
    label_output_path = os.path.join(output_dir, "labels", f"{base_filename}.txt")
    os.makedirs(os.path.dirname(label_output_path), exist_ok=True)
    try:
        with open(label_output_path, 'w') as f:
            f.write("\n".join(yolo_obbs))
        print(f"已保存标签到: {label_output_path}")
    except Exception as e:
        print(f"错误: 保存标签文件 {label_output_path} 时出错: {e}")
        return False


    # 复制图像文件
    if copy_images:
        # 确保目标图像文件名与标签文件名（不含扩展名）匹配
        dst_img_filename = f"{base_filename}{Path(image_filename).suffix}"
        image_output_path = os.path.join(output_dir, "images", dst_img_filename)
        os.makedirs(os.path.dirname(image_output_path), exist_ok=True)
        try:
            copy2(image_path, image_output_path)
        except Exception as e:
            print(f"错误: 复制图像 {image_path} 到 {image_output_path} 时出错: {e}")
    return True


def main(config_path):
    config = load_config(config_path)
    input_dir_path = config["input"] # 修改变量名以清晰表示这是目录
    output_root_dir = config["output"] # 修改变量名
    copy_images_flag = config["copy_images"] # 修改变量名
    raw_class_definitions = config["class_definitions"]

    if not os.path.exists(input_dir_path):
        print(f"错误: 输入路径 {input_dir_path} 不存在。")
        return

    # --- 创建类别名称到数字ID的映射 ---
    ordered_class_names = []
    class_name_to_id_map = {}
    processed_class_definitions_with_id = {} 
    current_numeric_id = 0

    for ann_id, definition in raw_class_definitions.items():
        class_name = definition["class_name"]
        if class_name not in class_name_to_id_map:
            class_name_to_id_map[class_name] = current_numeric_id
            ordered_class_names.append(class_name)
            numeric_id_for_class = current_numeric_id
            current_numeric_id += 1
        else:
            numeric_id_for_class = class_name_to_id_map[class_name]
        
        processed_class_definitions_with_id[ann_id] = {
            "class_id": numeric_id_for_class, # 存储数字ID
            "indices": definition["indices"]
        }
    # --- 映射创建完毕 ---

    processed_files_count = 0
    failed_files_count = 0

    if os.path.isfile(input_dir_path): # 虽然变量名是input_dir_path，但还是检查下万一是文件
        if input_dir_path.endswith(".json"):
            print(f"处理单个JSON文件: {input_dir_path}")
            if process_json_file(input_dir_path, output_root_dir, processed_class_definitions_with_id, copy_images_flag):
                processed_files_count +=1
            else:
                failed_files_count +=1
        else:
            print(f"错误: 输入文件 {input_dir_path} 不是JSON文件。")
            return
    elif os.path.isdir(input_dir_path):
        print(f"处理目录: {input_dir_path}")
        for root, _, files in os.walk(input_dir_path):
            for file in files:
                if file.endswith(".json"):
                    json_file_path = os.path.join(root, file)
                    print(f"处理: {json_file_path}")
                    if process_json_file(json_file_path, output_root_dir, processed_class_definitions_with_id, copy_images_flag):
                        processed_files_count += 1
                    else:
                        failed_files_count += 1
    else:
        print(f"错误: 输入路径 {input_dir_path} 既不是文件也不是目录。")
        return

    # --- 生成 data.yaml ---
    if processed_files_count > 0 : # 仅当成功处理了文件才生成yaml
        yaml_output_path = os.path.join(output_root_dir, "data.yaml")
        os.makedirs(os.path.dirname(yaml_output_path), exist_ok=True) # 确保输出目录存在
        try:
            with open(yaml_output_path, 'w') as f:
                f.write(f"train: ./images\n")
                f.write(f"val: ./images\n") # 假设训练和验证集相同
                f.write(f"test: # 可选\n\n")
                f.write(f"nc: {len(ordered_class_names)}\n")
                f.write(f"names:\n")
                for i, name in enumerate(ordered_class_names):
                    f.write(f"  {i}: {name}\n") # YOLO data.yaml 通常使用 id: name 格式
            print(f"已生成 data.yaml 到: {yaml_output_path}")
        except Exception as e:
            print(f"错误: 生成 data.yaml 时出错: {e}")
    else:
        print("未成功处理任何文件，不生成 data.yaml。")
    
    print(f"\n处理完毕。成功处理 {processed_files_count} 个文件，失败 {failed_files_count} 个文件。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将 Unity Perception JSON 文件转换为 YOLO OBB 标签格式。")
    parser.add_argument("config", help="配置文件的路径。")
    args = parser.parse_args()
    main(args.config)