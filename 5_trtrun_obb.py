# -*- coding: utf-8 -*-
"""YOLO OBB TensorRT 推理：仅强制 Resize，不使用 LetterBox。

与 4_onnxrun_obb.py 的输入输出完全一致：
    输入: uint8 BGR [1, 3, IMGSZ, IMGSZ]，外部负责 resize。
    输出: float32 [1, topk, 10]，每行 [x1, y1, x2, y2, x3, y3, x4, y4, confidence, class_id]。
"""

import glob
import os

import cv2
import numpy as np
import pycuda.autoinit  # noqa: F401  导入即初始化 CUDA 上下文
import pycuda.driver as cuda
import tensorrt as trt
from tqdm import tqdm

# 配置
ENGINE_PATH = "trt/best-smi.engine"
IMAGES_DIR = "images"
SAVE_DIR = "results_trt_obb"
IMGSZ = 800  # 必须与导出 engine 时的 imgsz 一致
# 置信度过滤已在导出时烘进计算图，低于 conf 的固定槽位输出 class_id = -1，此处无需再设阈值。

COLORS = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0), (255, 0, 255)]
IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")


class TRTRunner:
    """固定 batch=1 的 TensorRT engine 封装，显存只在构造时分配一次。"""

    def __init__(self, engine_path: str, imgsz: int) -> None:
        logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, "rb") as file, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(file.read())
        if self.engine is None:
            raise RuntimeError(f"engine 反序列化失败：{engine_path}（TensorRT 版本或 GPU 架构不匹配）")
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        modes = [self.engine.get_tensor_mode(name) for name in names]
        self.input_name = names[modes.index(trt.TensorIOMode.INPUT)]
        self.output_name = names[modes.index(trt.TensorIOMode.OUTPUT)]

        # 动态 batch 的 engine 必须先定形，否则拿不到输出的具体 shape。
        input_shape = (1, 3, imgsz, imgsz)
        if -1 in tuple(self.engine.get_tensor_shape(self.input_name)):
            self.context.set_input_shape(self.input_name, input_shape)
        self.input_shape = input_shape
        self.output_shape = tuple(self.context.get_tensor_shape(self.output_name))

        self.input_dtype = trt.nptype(self.engine.get_tensor_dtype(self.input_name))
        self.output_dtype = trt.nptype(self.engine.get_tensor_dtype(self.output_name))

        # 锁页主机内存 + 设备显存各分配一次，循环内只做拷贝。
        self.host_input = cuda.pagelocked_empty(int(np.prod(input_shape)), self.input_dtype)
        self.host_output = cuda.pagelocked_empty(int(np.prod(self.output_shape)), self.output_dtype)
        self.device_input = cuda.mem_alloc(self.host_input.nbytes)
        self.device_output = cuda.mem_alloc(self.host_output.nbytes)
        self.context.set_tensor_address(self.input_name, int(self.device_input))
        self.context.set_tensor_address(self.output_name, int(self.device_output))

    def infer(self, image_chw: np.ndarray) -> np.ndarray:
        """输入 uint8 BGR [1, 3, H, W]，返回 [topk, 10] 的检测结果。"""
        self.host_input[:] = image_chw.ravel()
        cuda.memcpy_htod_async(self.device_input, self.host_input, self.stream)
        if not self.context.execute_async_v3(stream_handle=self.stream.handle):
            raise RuntimeError("TensorRT 推理执行失败。")
        cuda.memcpy_dtoh_async(self.host_output, self.device_output, self.stream)
        self.stream.synchronize()
        return self.host_output.reshape(self.output_shape)[0]


def main():
    if not os.path.isfile(ENGINE_PATH):
        raise FileNotFoundError(f"未找到 TensorRT engine：{ENGINE_PATH}")

    os.makedirs(SAVE_DIR, exist_ok=True)
    image_paths = sorted(path for ext in IMAGE_EXTENSIONS for path in glob.glob(os.path.join(IMAGES_DIR, ext)))
    if not image_paths:
        raise FileNotFoundError(f"{IMAGES_DIR} 中未找到图片。")

    runner = TRTRunner(ENGINE_PATH, IMGSZ)
    print(f"模型：{ENGINE_PATH}，图片数量：{len(image_paths)}")
    print(f"输入：{runner.input_name} {runner.input_shape}，输出：{runner.output_name} {runner.output_shape}")

    for image_path in tqdm(image_paths):
        image = cv2.imread(image_path)  # OpenCV 读取为 BGR，正好匹配 engine 输入
        if image is None:
            continue
        original_h, original_w = image.shape[:2]
        resized = cv2.resize(image, (IMGSZ, IMGSZ), interpolation=cv2.INTER_LINEAR)
        input_data = np.ascontiguousarray(resized.transpose(2, 0, 1)[None], dtype=np.uint8)

        # 每行: [x1, y1, x2, y2, x3, y3, x4, y4, confidence, class_id]
        detections = runner.infer(input_data)
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
            cv2.putText(image, f"{class_id}:{score:.2f}", (x, max(20, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imwrite(os.path.join(SAVE_DIR, os.path.basename(image_path)), image)

    print(f"完成，结果保存至：{os.path.abspath(SAVE_DIR)}")


if __name__ == "__main__":
    main()
