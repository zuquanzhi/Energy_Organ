# 使用方法：python solo_parse.py sequence.1/step0.frame_data.json

import json
import cv2
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np

# 用于存储当前点击注释的全局变量
current_annotation = None


def draw_keypoints_interactive(json_path):
    """
    解析 Unity Perception JSON 文件，加载相应图像，并使用 matplotlib
    交互式地显示关键点及其坐标。

    Args:
        json_path (str): JSON 文件的路径。
    """
    global current_annotation
    # 检查 JSON 文件是否存在
    if not os.path.exists(json_path):
        print(f"错误：找不到 JSON 文件：{json_path}")
        return

    json_dir = os.path.dirname(json_path)

    # 加载 JSON 数据
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"错误：无法解码 JSON 文件：{json_path}")
        return
    except Exception as e:
        print(f"加载 JSON 文件时出错：{e}")
        return

    image_filename = None
    keypoints_data = []  # 存储关键点信息 (x, y, state, loc3d)

    # 查找图像文件名和关键点注释
    for capture in data.get('captures', []):
        if capture.get('@type') == "type.unity.com/unity.solo.RGBCamera":
            image_filename = capture.get('filename')
            # 收集此相机捕获的所有关键点注释
            for annotation in capture.get('annotations', []):
                if annotation.get('@type') == "type.unity.com/unity.solo.KeypointAnnotation":
                    for instance in annotation.get('values', []):
                        for keypoint in instance.get('keypoints', []):
                            state = keypoint.get('state', 0)
                            location = keypoint.get('location')
                            loc3d = keypoint.get('cameraCartesianLocation')
                            # 仅存储状态为 1 或 2 且位置有效的数据
                            if state in [1, 2] and location and len(location) == 2:
                                try:
                                    x, y = float(location[0]), float(
                                        location[1])
                                    # 确保 loc3d 存在且是列表或元组
                                    if not isinstance(loc3d, (list, tuple)):
                                        # 如果无效则设为 NaN
                                        loc3d = [np.nan, np.nan, np.nan]
                                    elif len(loc3d) != 3:
                                        # 如果长度不对，也标记为无效 (或尝试补齐/截断)
                                        print(
                                            f"警告: 无效的 3D 坐标长度 {loc3d}, 期望长度为 3。")
                                        loc3d = [np.nan, np.nan, np.nan]

                                    keypoints_data.append({
                                        'x': x,
                                        'y': y,
                                        'state': state,
                                        'loc3d': loc3d
                                    })
                                except (ValueError, TypeError):
                                    print(
                                        f"警告：跳过无效的关键点位置或状态：loc={location}, state={state}")
            # 假设每个 JSON 文件只有一个 RGB 相机捕获
            break

    if not image_filename:
        print("错误：在 JSON 文件中找不到 RGB 相机捕获或文件名。")
        return

    if not keypoints_data:
        print("警告：在 JSON 文件中找不到有效的关键点数据。")
        # return # 如果没有关键点，可以选择退出或只显示图像

    image_path = os.path.join(json_dir, image_filename)

    # 检查图像文件是否存在
    if not os.path.exists(image_path):
        print(f"错误：找不到图像文件：{image_path}")
        return

    # 使用 OpenCV 加载图像，然后转换为 RGB 以便 matplotlib 显示
    try:
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            raise ValueError("cv2.imread 返回 None，请检查文件路径和格式。")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"加载或转换图像时出错 {image_path}: {e}")
        return

    # 定义关键点状态的颜色 (RGB)
    colors_map = {
        1: 'red',   # 遮挡/未标记
        2: 'lime'   # 可见/已标记 (使用亮绿色以便区分)
    }
    radius = 5  # 散点的半径大小 (可以通过 s 参数控制)

    # 创建 matplotlib 图形和轴
    fig, ax = plt.subplots(figsize=(10, 8))  # 可以调整图形大小
    ax.imshow(image_rgb)
    ax.set_title(f"点击关键点查看坐标 - {os.path.basename(json_path)}")
    ax.axis('off')  # 关闭坐标轴显示

    # 提取坐标和颜色用于绘图
    points_x = [kp['x'] for kp in keypoints_data]
    points_y = [kp['y'] for kp in keypoints_data]
    point_colors = [colors_map[kp['state']] for kp in keypoints_data]

    # 使用 scatter 绘制关键点
    scatter = ax.scatter(points_x, points_y, c=point_colors,
                         s=radius**2, picker=True, pickradius=radius)

    # 定义点击事件处理函数
    def on_pick(event):
        global current_annotation
        # 如果之前有注释，先移除
        if current_annotation:
            current_annotation.remove()
            current_annotation = None

        if event.artist != scatter:
            # print("Clicked outside points")
            fig.canvas.draw_idle()  # 更新画布以移除旧注释
            return

        # 获取被点击点的索引
        ind = event.ind
        if not len(ind):
            return

        # 获取第一个被点击点的索引 (可能同时选中多个)
        point_index = ind[0]
        clicked_kp = keypoints_data[point_index]

        # 获取坐标
        x2d, y2d = clicked_kp['x'], clicked_kp['y']
        loc3d = clicked_kp['loc3d']
        state = clicked_kp['state']

        # 准备要显示的文本
        coord_text = (f"2D: ({x2d:.1f}, {y2d:.1f})\n"
                      f"3D: ({loc3d[0]:.3f}, {loc3d[1]:.3f}, {loc3d[2]:.3f})\n"
                      f"State: {state}")

        # 在点击位置附近显示注释
        current_annotation = ax.annotate(coord_text, (x2d, y2d),
                                         xytext=(10, 10), textcoords='offset points',
                                         bbox=dict(
                                             boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                                         arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

        # 在控制台也打印信息
        print(f"点击点索引: {point_index}, {coord_text.replace(chr(10), ' ')}")
        fig.canvas.draw_idle()  # 更新画布以显示新注释

    # 连接点击事件
    fig.canvas.mpl_connect('pick_event', on_pick)

    print(f"绘制了 {len(keypoints_data)} 个关键点。请在图像窗口中点击关键点。")
    plt.show()


if __name__ == "__main__":
    # 设置命令行参数解析器
    parser = argparse.ArgumentParser(
        description="从 Unity Perception JSON 文件中解析关键点并将其交互式地绘制到相应的图像上。")
    parser.add_argument("json_file", help="要处理的 stepX.frame_data.json 文件的路径。")

    # 解析参数
    args = parser.parse_args()

    # 调用主函数
    draw_keypoints_interactive(args.json_file)
