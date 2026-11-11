# -*- coding: utf-8 -*-
"""YOLO OBB ONNX 推理：仅强制 Resize，不使用 LetterBox。"""

import glob
import os

import cv2
import onnxruntime as ort
import numpy as np
from tqdm import tqdm

# 配置
ONNX_PATH = "onnx/best_obb_1tomany_nms.onnx"
# IMAGES_DIR = "images"
IMAGES_DIR = r"D:\BaiduNetdiskDownload\ICDAR2015\ch4_test_images"
SAVE_DIR = "results_onnx_obb"
IMGSZ = 800  # 必须与导出 ONNX 时的 --imgsz 一致
# 置信度过滤已在导出时烘进 ONNX 图，低于 --conf 的固定槽位输出 class_id = -1，此处无需再设阈值。

COLORS = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0), (255, 0, 255)]
IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")


def main():
    if not os.path.isfile(ONNX_PATH):
        raise FileNotFoundError(f"未找到 ONNX 模型：{ONNX_PATH}")

    os.makedirs(SAVE_DIR, exist_ok=True)
    image_paths = sorted(path for ext in IMAGE_EXTENSIONS for path in glob.glob(os.path.join(IMAGES_DIR, ext)))
    if not image_paths:
        raise FileNotFoundError(f"{IMAGES_DIR} 中未找到图片。")

    session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    input_name, output_name = session.get_inputs()[0].name, session.get_outputs()[0].name
    print(f"模型：{ONNX_PATH}，图片数量：{len(image_paths)}")

    for image_path in tqdm(image_paths):
        image = cv2.imread(image_path)  # OpenCV 读取为 BGR，正好匹配 ONNX 输入
        if image is None:
            continue
        original_h, original_w = image.shape[:2]
        resized = cv2.resize(image, (IMGSZ, IMGSZ), interpolation=cv2.INTER_LINEAR)
        input_data = np.ascontiguousarray(resized.transpose(2, 0, 1)[None], dtype=np.uint8)

        # 每行: [x1, y1, x2, y2, x3, y3, x4, y4, confidence, class_id]
        detections = session.run([output_name], {input_name: input_data})[0][0]
        for detection in detections:
            score, class_id = float(detection[8]), int(detection[9])
            if class_id < 0:  # 无效槽位：已被模型内的 conf 阈值或 NMS 过滤掉
                continue

            points = detection[:8].reshape(4, 2).copy()
            points[:, 0] *= original_w / IMGSZ  # Resize 后的 x 坐标还原至原图
            points[:, 1] *= original_h / IMGSZ  # Resize 后的 y 坐标还原至原图
            points = np.round(points).astype(np.int32).reshape(-1, 1, 2)
            color = COLORS[class_id % len(COLORS)]
            cv2.polylines(image, [points], isClosed=True, color=color, thickness=2)
            x, y = points[0, 0]
            # cv2.putText(image, f"{class_id}:{score:.4f}", (x, max(20, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
            cv2.putText(image, f"{score:.4f}", (x, max(20, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

        cv2.imwrite(os.path.join(SAVE_DIR, os.path.basename(image_path)), image)

    print(f"完成，结果保存至：{os.path.abspath(SAVE_DIR)}")


if __name__ == "__main__":
    main()
