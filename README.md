# YOLO26-OBB 全流程旋转目标检测与部署工具箱 (zlmaster)

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNXRuntime-Supported-brightgreen.svg)](https://onnxruntime.ai/)
[![TensorRT](https://img.shields.io/badge/TensorRT-Ready-green.svg)](https://developer.nvidia.com/tensorrt)
[![License: GPL-3.0](https://img.shields.io/badge/License-GPL--3.0-lightgrey.svg)](LICENSE)

本项目基于 **YOLO26 旋转目标检测 (Oriented Bounding Box, OBB)** 构建，提供了一套涵盖 **模型训练、双分支验证、预处理内嵌导出、ONNXRuntime 极简推理、TensorRT 高性能加速** 的完整工程化解决方案。

---

## 🌟 核心亮点与特性

1. **双检测头协同训练与独立最佳模型追踪 (Dual-Head)**：
   - 深入挖掘 YOLO26 的 **One-to-One (End2End / NMS-Free)** 与 **One-to-Many (依赖 NMS)** 双分支。
   - 训练期间支持固定轮次间隔评估 (`VALIDATE_EVERY = 5`)，并能**在同一个验证轮中分别评估双分支**，各自记录并导出最佳轮次权重（`best_epochXXX_one2one*.pt` / `best_epochXXX_one2many*.pt`）。
2. **免外部依赖的端到端 ONNX 导出**：
   - **内置 BGR 预处理**：模型直接接收 `uint8 BGR [B, 3, H, W]` 格式，内部自动完成 `float32` 转换、归一化与通道重排，外部仅需常规 `cv2.resize`，无需繁杂的 LetterBox 填充与逆变换。
   - **四顶点直接输出**：输出直接解析为 `[B, topk, 10]`（8 个顶点坐标 `[x1, y1, x2, y2, x3, y3, x4, y4]` + 置信度 + 类别 ID）。
   - **纯算子旋转 Fast-NMS**：针对 One-to-Many 分支，纯 PyTorch/ONNX 原生算子实现了**类别感知旋转 ProbIoU Fast-NMS**，并引入坐标尺度缩放（scale=32.0），彻底根除 TensorRT FP16 下 ProbIoU 计算易出现 NaN 溢出的顽疾。
3. **零门槛跨平台极简部署**：
   - 无需编译任何带有 C++/CUDA 外部依赖的旋转 NMS 算子插件（如 MMCV、RotatedNMS 插件等）。
   - 提供基于 **ONNXRuntime** 与 **TensorRT (PyCUDA)** 的开箱即用推理脚本，支持多边形绘制与结果保存。

---

## 📂 项目结构

```text
YOLO26_OBB_zlmaster/
├── 1_train.py                                 # 训练脚本（支持间隔验证、双分支指标分别跟踪）
├── 2_val.py                                   # PyTorch 原生批量推理/效果验证（输出可视化图/JSON/TXT）
├── 3_export_obb_带预处理_End2End.py            # 导出 One-to-One 端到端 ONNX（NMS-Free）
├── 3_export_obb_带预处理_OneToMany_RotatedNMS.py # 导出 One-to-Many ONNX（内嵌原生旋转 Fast-NMS）
├── 4_onnxrun_obb.py                           # ONNXRuntime 推理示例（直接对接 OpenCV 读取的图像）
├── 5_trtrun_obb.py                            # TensorRT 异步推理示例（锁页内存与显存复用）
├── ultralytics/                               # YOLO26 底层框架源码与网络结构定义
├── premodel/                                  # 官方预训练权重存放目录（如 yolo26n-obb.pt, yolo26s-obb.pt）
├── pt/                                        # 自定义微调后的模型权重（如 best.pt）
├── onnx/                                      # 导出的 ONNX 模型文件
├── trt/                                       # 编译生成的 TensorRT Engine 文件
└── images/                                    # 测试图片存放目录
```

---

## 🛠️ 环境准备

建议在 Anaconda / Miniconda 创建独立的 Python 虚拟环境：

```bash
conda create -n yolo26_obb python=3.10 -y
conda activate yolo26_obb

# 1. 安装 PyTorch (根据你的 CUDA 版本选择)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 2. 安装基础依赖
pip install opencv-python onnx onnxruntime tqdm pyyaml matplotlib pillow

# 3. (可选) 如需使用 TensorRT 加速
pip install tensorrt pycuda
```

---

## 🚀 快速上手流程

### 第一步：准备数据集与预训练模型

1. **预训练权重**：将官方预训练权重放入 `premodel/` 目录（例如 `yolo26n-obb.pt`、`yolo26s-obb.pt`）。
2. **OBB 数据集标注格式**：
   每行一个目标，坐标全部归一化至 `0 ~ 1`：
   ```text
   class_index x1 y1 x2 y2 x3 y3 x4 y4
   ```
3. **数据集 YAML 配置**：
   参考 `ultralytics/cfg/datasets/dota8.yaml`：
   ```yaml
   path: /path/to/dataset  # 数据集根目录
   train: images/train     # 训练图片 (相对 path)
   val: images/val         # 验证图片 (相对 path)
   names:
     0: plane
     1: ship
     2: storage-tank
   ```

---

### 第二步：模型训练 (`1_train.py`)

运行以下命令启动训练：

```bash
python 1_train.py
```

**训练特性说明**：
- **验证调度**：默认 `VALIDATE_EVERY = 5`（每 5 轮与最后一轮验证），节省遥感/高分辨率 OBB 验证时间。
- **分支跟踪**：自动同时评测 `one2one` 与 `one2many` 分支，当任一分支刷新最高指标时，自动导出精简体积的权重快照（如 `best_epoch035_one2one0.7852.pt`）。
- **无 Polars 兼容补丁**：已内置 `patch_results_csv_reader()`，修复了源码版保存检查点时因缺失 `polars` 导致的异常。
- **参数微调**：可在 `1_train.py` 中的 `TRAIN_ARGS` 自行调整 `imgsz`、`batch`、`epochs`、`degrees` 旋转数据增强等参数。

---

### 第三步：批量推理与验证 (`2_val.py`)

使用训练好的权重对 `images/` 目录下的测试图像进行批量推理：

```bash
python 2_val.py
```

**产物说明**（默认保存在 `runs/obb/predict/`）：
- `*.jpg`：绘制了类别、置信度以及旋转矩形框的可视化图片。
- `labels/*.txt`：标准 YOLO OBB 预测标签。
- `predictions.json`：包含每张图原图像素坐标系下的 `center_x, center_y, width, height, angle_radian, angle_degree` 及 `polygon` 四顶点坐标。

---

### 第四步：模型导出为 ONNX

根据业务场景需求，项目提供了两种导出模式：

#### 方案 A：导出 End-to-End ONNX (NMS-Free，推荐)
适合追求极简部署、低延迟的场景，直接利用 YOLO26 One-to-One 预测头：

```bash
python 3_export_obb_带预处理_End2End.py \
    --weights pt/best.pt \
    --outpath onnx/best_obb.onnx \
    --imgsz 800 \
    --topk 150 \
    --conf 0.25
```

#### 方案 B：导出 One-to-Many + 旋转 Fast-NMS ONNX
适合需要传统多重候选框过滤精度的场景，内嵌 ProbIoU 旋转 Fast-NMS：

```bash
python 3_export_obb_带预处理_OneToMany_RotatedNMS.py \
    --weights pt/best.pt \
    --outpath onnx/best_obb_1tomany_nms.onnx \
    --imgsz 800 \
    --topk 150 \
    --conf 0.25 \
    --iou 0.45
```

> **输入/输出规格说明**：
> - **Input**：`input` -> `shape: [B, 3, imgsz, imgsz]`, `dtype: uint8` (OpenCV BGR 格式)。
> - **Output**：`output` -> `shape: [B, topk, 10]`, `dtype: float32`。
> - 每行包含 10 项：`[x1, y1, x2, y2, x3, y3, x4, y4, confidence, class_id]`。
> - 无效或被过滤槽位：置信度为 0 且 `class_id = -1`。

---

### 第五步：ONNXRuntime 极简推理 (`4_onnxrun_obb.py`)

无需安装复杂的 C++ 插件，直接运行：

```bash
python 4_onnxrun_obb.py
```

推理逻辑极其简洁：
```python
# 1. 直接 cv2.resize 到模型要求尺寸
resized = cv2.resize(image, (IMGSZ, IMGSZ))
input_data = np.ascontiguousarray(resized.transpose(2, 0, 1)[None], dtype=np.uint8)

# 2. 推理获取 [topk, 10]
detections = session.run([output_name], {input_name: input_data})[0][0]

# 3. 按原始宽高比例缩放顶点坐标并绘制
for det in detections:
    if det[9] < 0: continue  # 过滤无效槽位
    pts = det[:8].reshape(4, 2)
    pts[:, 0] *= original_w / IMGSZ
    pts[:, 1] *= original_h / IMGSZ
```
检测可视化结果将保存至 `results_onnx_obb/`。

---

### 第六步：TensorRT 高性能加速部署 (`5_trtrun_obb.py`)

#### 1. 将 ONNX 编译为 TensorRT Engine
使用 NVIDIA 官方 `trtexec` 工具编译（支持 FP16 加速）：

```bash
trtexec --onnx=onnx/best_obb.onnx \
        --saveEngine=trt/best-smi.engine \
        --fp16
```

#### 2. 执行 TensorRT 推理
```bash
python 5_trtrun_obb.py
```
- 使用锁页主机内存 (Page-locked host memory) 和设备显存进行零重分配复用。
- 采用异步拷贝与流同步执行推理，结果将自动保存至 `results_trt_obb/`。

---

## 📌 常见问题与踩坑指南 (FAQ)

1. **Windows 下训练报错 `RuntimeError: DataLoader worker (pid ...) is killed by signal`**：
   - Windows 对多进程支持不同于 Linux，若发生内存或多线程报错，请在 `1_train.py` 中将 `workers=8` 调小（如 `workers=2` 或 `workers=0`）。
2. **TensorRT FP16 模式下 ProbIoU 出现 NaN**：
   - 浮点数计算距离与对数时若输入范围过大容易引发溢出，本项目在 `3_export_obb_带预处理_OneToMany_RotatedNMS.py` 中通过 `scale = 32.0` 对坐标进行数值缩放，有效解决了这一问题。
3. **模型输入为什么不需要 LetterBox**：
   - 遥感与工业检测场景中多为大分辨率切片图（如 800×800 或 1024×1024），直接 `cv2.resize` 能极大地简化 C++/Python 前后端工程流水线。如输入长宽比极端不一致，可自行保持等比例缩放并在后处理做坐标平移还原。

---

## 📄 开源许可

本项目遵循 [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.html)。欢迎 Issue 与 PR！
