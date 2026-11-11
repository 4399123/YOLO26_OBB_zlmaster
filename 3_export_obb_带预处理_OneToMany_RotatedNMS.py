# -*- coding: utf-8 -*-
"""导出 One-to-Many YOLO26-OBB ONNX，内嵌 BGR 预处理和类别感知旋转 Fast-NMS。

输入:  input  uint8 BGR [B, 3, imgsz, imgsz]。外部必须自行 resize，不包含 LetterBox。
输出:  output float32 [B, topk, 10]，每行是
       [x1, y1, x2, y2, x3, y3, x4, y4, confidence, class_id]。
       无效槽位为 [0, ..., 0, -1]。

示例:
    python 3_export_obb_带预处理_OneToMany_RotatedNMS.py --topk 150 --conf 0.25 --iou 0.45 --imgsz 800
"""

from __future__ import annotations

import argparse
import os

import onnx
import torch
import torch.nn as nn
from ultralytics import YOLO


class OBBOneToManyNMSWrapper(nn.Module):
    """BGR 输入、One-to-Many OBB 解码结果、类别感知旋转 Fast-NMS 和四顶点输出。"""

    def __init__(self, model: nn.Module, num_classes: int, topk: int, conf: float, iou: float) -> None:
        super().__init__()
        if topk < 1:
            raise ValueError("topk 必须大于 0。")
        if not 0.0 <= conf <= 1.0 or not 0.0 <= iou <= 1.0:
            raise ValueError("conf 和 iou 必须位于 0 到 1 之间。")
        self.model, self.num_classes = model, int(num_classes)
        self.topk, self.conf, self.iou = int(topk), float(conf), float(iou)
        self.pre_nms_topk = self.topk * 5  # 与 Ultralytics 导出包装器保持相同的候选预筛选倍率

    @staticmethod
    def xywhr_to_corners(xywhr: torch.Tensor) -> torch.Tensor:
        """将 [cx, cy, w, h, angle] 转为扁平四顶点 [x1, y1, ..., x4, y4]。"""
        center, width, height, angle = xywhr[..., :2], xywhr[..., 2:3], xywhr[..., 3:4], xywhr[..., 4:5]
        cos_angle, sin_angle = torch.cos(angle), torch.sin(angle)
        vector_w = torch.cat((width * cos_angle / 2, width * sin_angle / 2), dim=-1)
        vector_h = torch.cat((-height * sin_angle / 2, height * cos_angle / 2), dim=-1)
        corners = torch.stack(
            (center + vector_w + vector_h, center + vector_w - vector_h,
             center - vector_w - vector_h, center - vector_w + vector_h), dim=-2,
        )
        return corners.flatten(start_dim=-2)

    @staticmethod
    def rotated_probiou(boxes: torch.Tensor, scale: float = 32.0, eps: float = 1e-4) -> torch.Tensor:
        """计算批量 OBB ProbIoU，将空间坐标按 scale 缩放以防止 TensorRT FP16 溢出为 NaN。"""
        x, y, width, height, angle = (boxes[..., i : i + 1] for i in range(5))
        x, y, width, height = x / scale, y / scale, width / scale, height / scale
        cos_angle, sin_angle = torch.cos(angle), torch.sin(angle)
        a = ((width * cos_angle).square() + (height * sin_angle).square()) / 12
        b = ((width * sin_angle).square() + (height * cos_angle).square()) / 12
        c = width * height * torch.sin(2 * angle) / 24
        denominator = a + a.transpose(1, 2)
        denominator_b = b + b.transpose(1, 2)
        denominator_c = c + c.transpose(1, 2)
        determinant = denominator * denominator_b - denominator_c.square()
        dx, dy = x - x.transpose(1, 2), y - y.transpose(1, 2)
        t1 = (denominator * dy.square() + denominator_b * dx.square()) / (determinant + eps) * 0.25
        t2 = denominator_c * dx * dy / (determinant + eps) * 0.5
        box_det = (a * b - c.square()).clamp_min(0)
        t3 = (determinant / (4 * (box_det * box_det.transpose(1, 2)).sqrt() + eps) + eps).log() * 0.5
        distance = (t1 + t2 + t3).clamp(eps, 100.0)
        return 1 - (1 - (-distance).exp() + eps).sqrt()


    def forward(self, image_bgr: torch.Tensor) -> torch.Tensor:
        """输入 uint8 BGR，输出经过 conf 和类别感知旋转 NMS 过滤的固定 Top-K OBB 记录。"""
        image_rgb = image_bgr.float()[:, [2, 1, 0], ...].div(255.0)
        predictions = self.model(image_rgb)
        if isinstance(predictions, (tuple, list)):
            predictions = predictions[0]

        # One-to-Many OBB 输出：[B, 4 + nc + 1, N] -> [B, N, 4 + nc + 1]。
        predictions = predictions.transpose(1, 2)
        boxes = predictions[..., :4]
        class_scores = predictions[..., 4 : 4 + self.num_classes]
        angle = predictions[..., 4 + self.num_classes : 5 + self.num_classes]
        confidence, class_id = torch.max(class_scores, dim=-1)

        # 先选取更多候选框，减少旋转 IoU 矩阵尺寸；低分候选的 score 设为 0。
        candidate_count = min(self.pre_nms_topk, confidence.shape[1])
        scores, indices = torch.topk(confidence, k=candidate_count, dim=1, largest=True, sorted=True)
        gather_boxes = indices.unsqueeze(-1).expand(-1, -1, 4)
        gather_angle = indices.unsqueeze(-1).expand(-1, -1, 1)
        boxes = torch.gather(boxes, 1, gather_boxes)
        angle = torch.gather(angle, 1, gather_angle)
        class_id = torch.gather(class_id, 1, indices)
        scores = scores * (scores >= self.conf).to(scores.dtype)
        xywhr = torch.cat((boxes, angle), dim=-1)

        # 类别感知旋转 Fast-NMS：仅同类框之间的 ProbIoU 才会抑制。
        probiou = self.rotated_probiou(xywhr)
        same_class = class_id.unsqueeze(2) == class_id.unsqueeze(1)
        count = scores.shape[1]
        row = torch.arange(count, device=scores.device).view(1, count, 1)
        column = torch.arange(count, device=scores.device).view(1, 1, count)
        upper_triangle = row < column  # 行是更高分框，列是待抑制的低分框
        suppress = ((probiou > self.iou) & same_class & upper_triangle).any(dim=1)
        scores = scores * (~suppress).to(scores.dtype)

        # 取最终 Top-K；被抑制或低于 conf 的记录在固定槽位中标记为无效。
        final_count = min(self.topk, scores.shape[1])
        scores, indices = torch.topk(scores, k=final_count, dim=1, largest=True, sorted=True)
        xywhr = torch.gather(xywhr, 1, indices.unsqueeze(-1).expand(-1, -1, 5))
        class_id = torch.gather(class_id, 1, indices).to(scores.dtype)
        output = torch.cat((self.xywhr_to_corners(xywhr), scores.unsqueeze(-1), class_id.unsqueeze(-1)), dim=-1)
        valid = scores.unsqueeze(-1) > 0
        invalid_record = torch.zeros_like(output)
        invalid_record[..., -1] = -1.0
        return torch.where(valid.expand_as(output), output, invalid_record)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出带 BGR 预处理和 One-to-Many 旋转 NMS 的 OBB ONNX 模型")
    parser.add_argument("--weights", default="pt/best.pt", help="训练完成的 .pt 权重")
    parser.add_argument("--outpath", default="onnx/best_obb_1tomany_nms.onnx", help="ONNX 输出路径")
    parser.add_argument("--topk", type=int, default=100, help="NMS 后每张图最多输出的有效候选框数量")
    parser.add_argument("--conf", type=float, default=0.55, help="类别置信度阈值")
    parser.add_argument("--iou", type=float, default=0.25, help="旋转 ProbIoU NMS 抑制阈值")
    parser.add_argument("--imgsz", type=int, default=800, help="模型输入宽高；外部应先 resize 到该尺寸")
    parser.add_argument("--opset", type=int, default=13, help="ONNX opset 版本，推荐 13")
    return parser.parse_args()


def main(weights: str, outpath: str, topk: int, conf: float, iou: float, imgsz: int, opset: int) -> None:
    if not os.path.isfile(weights):
        raise FileNotFoundError(f"未找到权重文件：{os.path.abspath(weights)}")
    if imgsz <= 0 or topk <= 0:
        raise ValueError("imgsz 和 topk 必须大于 0。")
    if not 0.0 <= conf <= 1.0 or not 0.0 <= iou <= 1.0:
        raise ValueError("conf 和 iou 必须位于 0 到 1 之间。")
    output_dir = os.path.dirname(outpath)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"加载 OBB 权重：{weights}")
    yolo = YOLO(weights)
    model = yolo.model
    model.end2end = False  # 必须在 fuse 之前切换，以保留 One-to-Many 检测头。
    model = model.fuse(verbose=False).eval()
    head = model.model[-1]
    if head.cv2 is None:
        raise RuntimeError("One-to-Many 检测头不可用；请确认在 model.fuse() 前设置了 model.end2end = False。")

    wrapper = OBBOneToManyNMSWrapper(model, int(head.nc), topk, conf, iou).eval()
    dummy_bgr = torch.zeros(1, 3, imgsz, imgsz, dtype=torch.uint8)
    with torch.inference_mode():
        sample_output = wrapper(dummy_bgr)

    print(f"导出 ONNX：{outpath}")
    torch.onnx.export(
        wrapper, dummy_bgr, outpath, opset_version=opset, input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}, do_constant_folding=True,
    )
    print("使用 onnxslim 优化并直接覆盖导出文件...")
    import onnxslim

    optimized_model = onnxslim.slim(onnx.load(outpath))
    onnx.checker.check_model(optimized_model)
    onnx.save(optimized_model, outpath)
    print("导出完成")
    print(f"ONNX 文件：{os.path.abspath(outpath)}")
    print(f"输入：input, uint8 BGR, [B, 3, {imgsz}, {imgsz}]")
    print(f"置信度阈值：{conf:.4f}；旋转 NMS ProbIoU 阈值：{iou:.4f}")
    print(f"输出：output, float32, [B, {sample_output.shape[1]}, 10]")
    print("输出字段：[x1, y1, x2, y2, x3, y3, x4, y4, confidence, class_id]")


if __name__ == "__main__":
    args = parse_args()
    main(args.weights, args.outpath, args.topk, args.conf, args.iou, args.imgsz, args.opset)
