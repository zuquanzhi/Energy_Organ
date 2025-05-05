# convert_unity_to_yolo_obb.py

import os
import json
import math
import argparse
import glob
import numpy as np
import cv2
import shutil
from pathlib import Path


def calculate_obb_from_keypoints(keypoints):
    """
    从可见关键点计算最小面积旋转框（OBB）
    返回 (xc, yc, w, h, angle_rad) 或 None
    """
    valid_kps = [(x, y) for x, y, v in keypoints if v > 0]
    if len(valid_kps) < 2:
        return None

    points = np.array(valid_kps, dtype=np.float32)

    try:
        rect = cv2.minAreaRect(points)
        box = cv2.boxPoints(rect)
        (xc, yc), (w, h), angle = rect
        angle_rad = math.radians(angle)
        w = max(w, 1e-6)
        h = max(h, 1e-6)
        return (xc, yc, w, h, angle_rad)
    except Exception as e:
        print(f"cv2.minAreaRect 计算失败: {e}")
        xs = [x for x, y in valid_kps]
        ys = [y for x, y in valid_kps]
        xc = (min(xs) + max(xs)) / 2
        yc = (min(ys) + max(ys)) / 2
        w = max(xs) - min(xs)
        h = max(ys) - min(ys)
        return (xc, yc, w, h, 0.0)


def extract_keypoints_data(json_path):
    """
    提取所有物体的关键点数据（支持多个 instanceId）
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"无法加载 JSON 文件: {json_path}, 错误: {e}")
        return None

    result = {
        "width": 0,
        "height": 0,
        "image_filename": "",
        "instances": [],
        "instance_id_to_class_id": {}
    }

    instance_ids = set()
    instance_counter = 0

    for capture in data.get('captures', []):
        if capture.get('@type') != "type.unity.com/unity.solo.RGBCamera":
            continue

        result["width"] = int(capture['dimension'][0])
        result["height"] = int(capture['dimension'][1])
        result["image_filename"] = capture.get('filename', '')

        for annotation in capture.get('annotations', []):
            if annotation.get('@type') != "type.unity.com/unity.solo.KeypointAnnotation":
                continue

            for instance in annotation.get('values', []):
                instance_id = instance.get('instanceId')
                label_id = instance.get('labelId', 0)
                keypoints = []

                for kp in instance.get('keypoints', []):
                    loc = kp.get('location')
                    state = kp.get('state', 0)

                    if loc and len(loc) == 2:
                        x = float(loc[0])
                        y = float(loc[1])
                        visibility = 2 if state == 2 else 1 if state == 1 else 0
                        keypoints.append((x, y, visibility))
                    else:
                        keypoints.append((0.0, 0.0, 0))

                if any(v > 0 for _, _, v in keypoints):
                    # 为每个物体实例生成一个唯一标识符
                    unique_id = f"{instance_id}_{instance_counter}"
                    instance_counter += 1
                    
                    instance_ids.add(unique_id)
                    result["instances"].append({
                        "instanceId": instance_id,
                        "uniqueId": unique_id,  # 添加唯一ID
                        "keypoints": keypoints
                    })

    # 构建 uniqueId -> class_id 映射（每个框一个 class_id）
    result["instance_id_to_class_id"] = {inst_id: idx for idx, inst_id in enumerate(sorted(instance_ids))}

    return result


def save_yolo_obb(keypoints_data, output_file):
    """
    保存为 YOLO OBB 格式：
    class_id xc yc w h angle_rad
    每个矩形框对应一个唯一的 class_id
    """
    if not keypoints_data or not keypoints_data['instances']:
        print(f"没有有效数据，跳过保存 {output_file}")
        return

    img_w = keypoints_data['width']
    img_h = keypoints_data['height']
    instance_id_to_class_id = keypoints_data.get("instance_id_to_class_id", {})

    with open(output_file, 'w') as f:
        for obj in keypoints_data['instances']:
            kps = obj['keypoints']
            obb_data = calculate_obb_from_keypoints(kps)
            if obb_data is None:
                continue

            xc, yc, w, h, angle = obb_data
            xc_norm = xc / img_w
            yc_norm = yc / img_h
            w_norm = w / img_w
            h_norm = h / img_h

            # 使用唯一ID而不是instanceId来获取class_id
            unique_id = obj['uniqueId']
            class_id = instance_id_to_class_id.get(unique_id, 0)

            f.write(f"{class_id} {xc_norm:.6f} {yc_norm:.6f} {w_norm:.6f} {h_norm:.6f} {angle:.6f}\n")


def copy_image_file(json_path, image_filename, output_dir):
    if not image_filename:
        return False

    json_dir = os.path.dirname(json_path)
    src_image_path = os.path.join(json_dir, image_filename)

    if not os.path.exists(src_image_path):
        print(f"找不到图像文件: {src_image_path}")
        return False

    dst_image_path = os.path.join(output_dir, os.path.basename(src_image_path))
    shutil.copy2(src_image_path, dst_image_path)
    print(f"已复制图像: {src_image_path} -> {dst_image_path}")
    return True


def process_single_file(json_path, output_obb_dir, copy_images=True):
    print(f"处理文件: {json_path}")
    data = extract_keypoints_data(json_path)
    if not data or not data['instances']:
        print(f"无法提取有效数据: {json_path}")
        return

    base_name = os.path.splitext(os.path.basename(json_path))[0]
    obb_file = os.path.join(output_obb_dir, f"{base_name}.txt")
    save_yolo_obb(data, obb_file)
    print(f"已保存 OBB 标签到: {obb_file}")

    if copy_images:
        image_output_dir = os.path.join(os.path.dirname(output_obb_dir), "images", "obb")
        os.makedirs(image_output_dir, exist_ok=True)
        copy_image_file(json_path, data.get("image_filename"), image_output_dir)


def process_directory(input_dir, output_obb_dir, copy_images=True):
    json_files = glob.glob(os.path.join(input_dir, "**/*frame_data.json"), recursive=True)
    if not json_files:
        print(f"在 {input_dir} 中未找到 frame_data.json 文件")
        return

    print(f"共找到 {len(json_files)} 个 JSON 文件，开始构建类别映射...")

    # 收集所有唯一ID
    all_unique_ids = set()
    all_data = []
    
    for json_file in json_files:
        data = extract_keypoints_data(json_file)
        if data and data['instances']:
            all_data.append((json_file, data))
            for obj in data['instances']:
                all_unique_ids.add(obj['uniqueId'])

    # 构建映射表
    instance_id_to_class_id = {unique_id: idx for idx, unique_id in enumerate(sorted(all_unique_ids))}
    print(f"构建完成，共 {len(instance_id_to_class_id)} 个唯一框")

    # 生成 data.yaml
    generate_yaml(instance_id_to_class_id, output_obb_dir)

    # 逐个处理文件
    for json_file, data in all_data:
        data["instance_id_to_class_id"] = instance_id_to_class_id
        base_name = os.path.splitext(os.path.basename(json_file))[0]
        obb_file = os.path.join(output_obb_dir, f"{base_name}.txt")
        save_yolo_obb(data, obb_file)
        print(f"已保存 OBB 标签到: {obb_file}")

        if copy_images:
            image_output_dir = os.path.join(os.path.dirname(output_obb_dir), "images", "obb")
            os.makedirs(image_output_dir, exist_ok=True)
            copy_image_file(json_file, data.get("image_filename"), image_output_dir)


def generate_yaml(instance_id_to_class_id, output_dir):
    names = [f"box_{unique_id}" for unique_id in sorted(instance_id_to_class_id.keys())]
    data = {
        'train': '../images/obb',
        'val': '../images/obb',
        'nc': len(names),
        'names': names
    }

    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, 'w') as f:
        for key, value in data.items():
            if isinstance(value, list):
                f.write(f"{key}:\n")
                for item in value:
                    f.write(f"  - {item}\n")
            else:
                f.write(f"{key}: {value}\n")
    print(f"已生成 data.yaml: {yaml_path}")


def main():
    parser = argparse.ArgumentParser(description="将 Unity JSON 转换为 YOLO OBB 格式，每个矩形框对应一个唯一ID")
    parser.add_argument("input", help="输入 JSON 文件路径或目录")
    parser.add_argument("--output-obb", required=True, help="输出 OBB 标签的目录")
    parser.add_argument("--copy-images", action="store_true", default=True, help="复制图像文件")
    parser.add_argument("--no-copy-images", dest="copy_images", action="store_false", help="不复制图像文件")

    args = parser.parse_args()

    output_obb_dir = os.path.abspath(args.output_obb)
    os.makedirs(output_obb_dir, exist_ok=True)

    if os.path.isfile(args.input):
        print("处理单个 JSON 文件")
        data = extract_keypoints_data(args.input)
        if not data:
            return

        # 处理单个文件的唯一ID映射已在extract_keypoints_data中完成
        process_single_file(args.input, output_obb_dir, args.copy_images)

    elif os.path.isdir(args.input):
        print("处理整个目录下的 JSON 文件")
        process_directory(args.input, output_obb_dir, args.copy_images)

    else:
        print(f"输入无效: {args.input}")


if __name__ == "__main__":
    main()