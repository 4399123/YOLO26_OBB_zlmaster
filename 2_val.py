# -*- coding: utf-8 -*-
"""YOLO OBB 批量推理与效果验证脚本。

默认使用 pt/best.pt，对 images/ 内全部图片进行旋转框检测。
输出目录默认为 runs/obb/predict，包含：
    - *.jpg：绘制了类别、置信度和旋转框的可视化结果。
    - labels/*.txt：YOLO OBB 格式预测结果，类别 + 归一化四顶点 + 置信度。
    - predictions.json：原图像素坐标、旋转框参数和四顶点坐标的完整结果。

用法：
    python 2_val.py
"""

from __future__ import annotations

import json
from pathlib import Path

from ultralytics import YOLO

# ======================================================================================================================
# 配置
# ======================================================================================================================
MODEL_PATH = Path("pt/best.pt")  # 训练好的 OBB 权重
SOURCE_DIR = Path("images")  # 待验证图片目录
PROJECT = "runs/obb"
NAME = "predict"

CONF = 0.25  # 置信度阈值：低一些召回更高，误检也可能增多
IOU = 0.7  # NMS IoU 阈值（YOLO26 端到端模型主要保留作兼容配置）
IMGSZ = 800  # 应与训练 imgsz 一致或接近
DEVICE = None  # None 自动选择；可指定 0、"cpu" 等
MAX_DET = 300  # 每张图最多保留的检测数量

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def get_images(source_dir: Path) -> list[Path]:
    """返回目录中的全部支持格式图像。"""
    return sorted(path for path in source_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def serialize_predictions(results) -> list[dict]:
    """将 Ultralytics OBB Results 转为适合保存的 JSON 数据。"""
    records: list[dict] = []
    for result in results:
        image_record = {"image": Path(result.path).name, "objects": []}
        if result.obb is None or len(result.obb) == 0:
            records.append(image_record)
            continue

        xywhr = result.obb.xywhr.cpu().numpy()
        corners = result.obb.xyxyxyxy.cpu().numpy()
        confidences = result.obb.conf.cpu().numpy()
        class_ids = result.obb.cls.cpu().numpy().astype(int)

        for box, polygon, confidence, class_id in zip(xywhr, corners, confidences, class_ids):
            cx, cy, width, height, angle_radian = (float(value) for value in box)
            image_record["objects"].append(
                {
                    "class_id": int(class_id),
                    "class_name": result.names[class_id],
                    "confidence": round(float(confidence), 6),
                    "xywhr": {
                        "center_x": round(cx, 3),
                        "center_y": round(cy, 3),
                        "width": round(width, 3),
                        "height": round(height, 3),
                        "angle_radian": round(angle_radian, 6),
                        "angle_degree": round(angle_radian * 180.0 / 3.141592653589793, 3),
                    },
                    "polygon": [[round(float(x), 3), round(float(y), 3)] for x, y in polygon],
                }
            )
        records.append(image_record)
    return records


def main() -> None:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"未找到训练权重：{MODEL_PATH.resolve()}")
    if not SOURCE_DIR.is_dir():
        raise FileNotFoundError(f"未找到测试图片目录：{SOURCE_DIR.resolve()}")

    images = get_images(SOURCE_DIR)
    if not images:
        raise FileNotFoundError(f"{SOURCE_DIR.resolve()} 中没有支持的图片文件。")

    print(f"加载模型：{MODEL_PATH}")
    print(f"待推理图像：{len(images)} 张，来源：{SOURCE_DIR}")

    model = YOLO(MODEL_PATH)
    results = model.predict(
        source=[str(path) for path in images],
        conf=CONF,
        iou=IOU,
        imgsz=IMGSZ,
        device=DEVICE,
        max_det=MAX_DET,
        save=True,  # 保存带旋转框的可视化图像
        save_txt=True,  # 保存归一化四点坐标的 YOLO OBB 标签
        save_conf=True,  # TXT 最后额外保存置信度
        project=PROJECT,
        name=NAME,
        exist_ok=True,  # 重复运行时覆盖同一输出目录中的同名结果
        verbose=True,
    )

    save_dir = Path(results[0].save_dir) if results else Path(PROJECT) / NAME
    prediction_records = serialize_predictions(results)
    json_path = save_dir / "predictions.json"
    json_path.write_text(
        json.dumps(
            {
                "model": str(MODEL_PATH),
                "source": str(SOURCE_DIR),
                "conf_threshold": CONF,
                "iou_threshold": IOU,
                "images": prediction_records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    total_objects = sum(len(record["objects"]) for record in prediction_records)
    print("\n推理完成")
    print(f"检测到目标总数：{total_objects}")
    print(f"可视化结果目录：{save_dir.resolve()}")
    print(f"YOLO OBB 标签目录：{(save_dir / 'labels').resolve()}")
    print(f"JSON 结果文件：{json_path.resolve()}")

    for record in prediction_records:
        print(f"  {record['image']}: {len(record['objects'])} 个目标")


if __name__ == "__main__":
    main()
