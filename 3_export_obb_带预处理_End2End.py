# -*- coding: utf-8 -*-
"""导出真正启用 One-to-One End-to-End 分支的 YOLO26-OBB ONNX 模型。

ONNX 输入:
    input: uint8 BGR, shape [B, 3, imgsz, imgsz]
    注意：不包含 resize / letterbox；外部必须预先将图像调整到 imgsz × imgsz。

ONNX 输出:
    output: float32, shape [B, topk, 10]
    每行: [x1, y1, x2, y2, x3, y3, x4, y4, confidence, class_id]
    坐标是输入图像坐标系下的像素值，不是原始未缩放图像坐标，也未归一化。
    置信度低于 conf 的固定槽位会输出 [0, ..., 0, -1]，即 confidence=0 且 class_id=-1。

用法:
    python 3_export_obb_带预处理.py
    python 3_export_obb_带预处理.py --weights pt/best.pt --outpath onnx/best_obb.onnx --topk 150 --imgsz 800 --conf 0.25
"""

from __future__ import annotations

import argparse
import os

import onnx
import torch
import torch.nn as nn
from ultralytics import YOLO


class OBBExportWrapper(nn.Module):
    """封装真正的 End-to-End OBB：BGR uint8 预处理、Top-K 和 xywhr 到四顶点转换。"""

    def __init__(self, model: nn.Module, num_classes: int, topk: int, conf: float) -> None:
        super().__init__()
        if topk < 1:
            raise ValueError("topk 必须大于 0。")
        if not 0.0 <= conf <= 1.0:
            raise ValueError("conf 必须位于 0 到 1 之间。")
        self.model = model
        self.num_classes = int(num_classes)
        self.topk = int(topk)
        self.conf = float(conf)

    @staticmethod
    def xywhr_to_corners(xywhr: torch.Tensor) -> torch.Tensor:
        """将 [cx, cy, w, h, angle] 转为 [x1, y1, ..., x4, y4]。"""
        center, width, height, angle = xywhr[..., :2], xywhr[..., 2:3], xywhr[..., 3:4], xywhr[..., 4:5]
        cos_angle, sin_angle = torch.cos(angle), torch.sin(angle)
        vector_w = torch.cat((width * cos_angle / 2, width * sin_angle / 2), dim=-1)
        vector_h = torch.cat((-height * sin_angle / 2, height * cos_angle / 2), dim=-1)
        corners = torch.stack(
            (center + vector_w + vector_h, center + vector_w - vector_h,
             center - vector_w - vector_h, center - vector_w + vector_h),
            dim=-2,
        )
        return corners.flatten(start_dim=-2)

    def forward(self, image_bgr: torch.Tensor) -> torch.Tensor:
        """输入 uint8 BGR [B, 3, H, W]，输出 End-to-End 模型的 Top-K OBB 四顶点、置信度和类别。"""
        image_rgb = image_bgr.float()[:, [2, 1, 0], ...].div(255.0)
        predictions = self.model(image_rgb)
        if isinstance(predictions, (tuple, list)):
            predictions = predictions[0]

        # 真正的 YOLO26 One-to-One / End-to-End 输出为 [B, N, 7]:
        # [cx, cy, w, h, confidence, class_id, angle_radian]。
        confidence = predictions[..., 4]
        k = min(self.topk, confidence.shape[1])
        selected_confidence, indices = torch.topk(confidence, k=k, dim=1, largest=True, sorted=True)
        selected_predictions = torch.gather(predictions, 1, indices.unsqueeze(-1).expand(-1, -1, 7))
        selected_xywhr = torch.cat((selected_predictions[..., :4], selected_predictions[..., 6:7]), dim=-1)
        selected_class_id = selected_predictions[..., 5:6]
        output = torch.cat((self.xywhr_to_corners(selected_xywhr), selected_confidence.unsqueeze(-1), selected_class_id), dim=-1)

        # End-to-End 分支本身不需要额外 NMS；仅将低置信度固定槽位标记为无效。
        valid = selected_confidence.unsqueeze(-1) >= self.conf
        invalid_record = torch.zeros_like(output)
        invalid_record[..., -1] = -1.0
        return torch.where(valid.expand_as(output), output, invalid_record)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出带 BGR 预处理与 OBB 四顶点输出的 ONNX 模型")
    parser.add_argument("--weights", default="pt/best.pt", help="训练完成的 .pt 权重")
    parser.add_argument("--outpath", default="onnx/best_obb_1to1.onnx", help="ONNX 输出路径")
    parser.add_argument("--topk", type=int, default=100, help="每张图输出置信度最高的候选框数量")
    parser.add_argument("--conf", type=float, default=0.55, help="类别置信度阈值；低于该值的输出行会标记为无效")
    parser.add_argument("--imgsz", type=int, default=800, help="模型输入宽高；外部应先 resize 到该尺寸")
    parser.add_argument("--opset", type=int, default=13, help="ONNX opset 版本，推荐 13")
    return parser.parse_args()


def main(weights: str, outpath: str, topk: int, conf: float, imgsz: int, opset: int) -> None:
    if not os.path.isfile(weights):
        raise FileNotFoundError(f"未找到权重文件：{os.path.abspath(weights)}")
    if imgsz <= 0:
        raise ValueError("imgsz 必须大于 0。")
    if topk <= 0:
        raise ValueError("topk 必须大于 0。")
    if not 0.0 <= conf <= 1.0:
        raise ValueError("conf 必须位于 0 到 1 之间。")

    output_dir = os.path.dirname(outpath)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    print(f"加载 OBB 权重：{weights}")
    yolo = YOLO(weights)
    model = yolo.model
    # 必须在 fuse() 之前启用 One-to-One 分支；否则会导出密集 One-to-Many 候选框且没有 NMS。
    model.end2end = True
    model = model.fuse(verbose=False).eval()
    if model.model[-1].cv2 is not None:
        raise RuntimeError("未成功启用 End-to-End 检测头。")
    num_classes = int(model.model[-1].nc)
    wrapper = OBBExportWrapper(model, num_classes=num_classes, topk=topk, conf=conf).eval()

    # 固定空间尺寸，因为外部负责 resize；仅 batch 维度可动态变化。
    dummy_bgr = torch.zeros(1, 3, imgsz, imgsz, dtype=torch.uint8)
    with torch.inference_mode():
        sample_output = wrapper(dummy_bgr)

    print(f"导出 ONNX：{outpath}")
    torch.onnx.export(
        wrapper,
        dummy_bgr,
        str(outpath),
        opset_version=opset,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        do_constant_folding=True,
    )

    print("使用 onnxslim 优化并直接覆盖导出文件...")
    import onnxslim

    optimized_model = onnxslim.slim(onnx.load(str(outpath)))
    onnx.checker.check_model(optimized_model)
    onnx.save(optimized_model, str(outpath))

    print("导出完成")
    print(f"ONNX 文件：{os.path.abspath(outpath)}")
    print(f"输入：input, uint8 BGR, [B, 3, {imgsz}, {imgsz}]")
    print(f"置信度阈值：{conf:.4f}；低于阈值的输出行为 [0, ..., 0, -1]")
    print(f"输出：output, float32, [B, {sample_output.shape[1]}, 10]")
    print("输出字段：[x1, y1, x2, y2, x3, y3, x4, y4, confidence, class_id]")


if __name__ == "__main__":
    args = parse_args()
    main(args.weights, args.outpath, args.topk, args.conf, args.imgsz, args.opset)
