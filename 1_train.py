# -*- coding: utf-8 -*-
"""
YOLO 旋转框 (OBB, Oriented Bounding Box) 目标检测训练脚本.

参考:
    - 官方训练文档: https://docs.ultralytics.com/zh/modes/train
    - 全部参数默认值: ultralytics/cfg/default.yaml
    - OBB 数据集格式示例: ultralytics/cfg/datasets/dota8.yaml

OBB 标签格式 (每行一个目标, 坐标归一化到 0~1):
    class_index x1 y1 x2 y2 x3 y3 x4 y4

数据集 yaml 示例:
    path: D:/datasets/my_obb   # 数据集根目录
    train: images/train        # 训练图片 (相对 path)
    val: images/val            # 验证图片 (相对 path)
    names:
      0: plane
      1: ship

用法:
    python 1_train.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import torch

from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import strip_optimizer

# 每隔多少轮在验证集计算一次精度。最后一轮始终会额外验证一次。
VALIDATE_EVERY = 5
BEST_METRIC_NAME = "metrics/mAP50-95(B)"  # OBB 的主评估指标

# YOLO26 有两个检测头: one2many 需配外部 NMS, one2one 是 NMS-free (End2End)。
# 每个验证轮会把两个分支各跑一遍，分别追踪各自最佳以及两者均值最佳的权重。
BRANCH_ONE2ONE = "one2one"
BRANCH_ONE2MANY = "one2many"
TRACK_TARGETS = (BRANCH_ONE2ONE, BRANCH_ONE2MANY, "mean")


def patch_results_csv_reader() -> None:
    """为当前源码版 Ultralytics 提供无 polars 的 results.csv 读取后备实现。"""
    # 当前仓库的 trainer.py 在保存权重时强制导入 polars；缺失时训练会在首个保存点中断。
    from ultralytics.engine.trainer import BaseTrainer

    def read_results_csv_without_polars(self):
        if not self.csv.exists():
            return {}
        with self.csv.open("r", encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))
        if not rows:
            return {}
        return {
            key: [float(row[key]) if row.get(key, "").strip() else None for row in rows]
            for key in rows[0]
        }

    BaseTrainer.read_results_csv = read_results_csv_without_polars


def evaluate_other_branch(trainer) -> float | None:
    """再跑一遍验证集，评估本轮未被框架验证的那个检测头，返回其 mAP50-95。"""
    # 框架已按 args.nms 验证过一个分支；这里临时翻转 end2end 再复用同一个 validator。
    validator, model = trainer.validator, trainer.ema.ema
    if not hasattr(model, "end2end"):
        return None

    saved_nms, saved_plots = validator.args.nms, validator.args.plots
    # 翻转 nms 即翻转分支：validator.__call__ 内部按 `args.nms is False` 设置 model.end2end。
    validator.args.nms = None if saved_nms is False else False
    validator.args.plots = False  # 第二遍不重复画图，也避免覆盖第一遍的产物
    try:
        metrics = validator(trainer)
    except Exception as error:  # 附加评估失败不应中断训练
        LOGGER.warning(f"附加分支验证失败，本轮跳过：{error}")
        return None
    finally:
        # 无论成败都要还原，否则后续轮次会验证错误的分支。
        validator.args.nms, validator.args.plots = saved_nms, saved_plots
        model.end2end = saved_nms is False
        # validator 把 loss/metrics 累加在自身状态上，重置以免污染框架记录的本轮结果。
        validator.loss = {k: torch.zeros_like(v) for k, v in trainer.loss_items.items()}

    return None if metrics is None else metrics.get(BEST_METRIC_NAME)


def validate_and_track_best(trainer) -> None:
    """仅每 VALIDATE_EVERY 轮验证；分别追踪 one2one、one2many 及两者均值的最佳权重。"""
    epoch = trainer.epoch + 1  # 框架内部 epoch 从 0 开始，本脚本显示从 1 开始
    # 训练结束后 final_eval() 会把 epoch 加 1 并再次触发本回调，用来复验 best.pt。
    # 那次的 metrics 来自 best.pt 而非当前 EMA，与 last.pt 不对应，必须跳过，否则会记录错误快照。
    if epoch > trainer.epochs:
        return

    is_final_epoch = epoch >= trainer.epochs
    if epoch % VALIDATE_EVERY != 0 and not is_final_epoch:
        trainer.args.val = False  # 下一轮不验证，由框架跳过精度计算
        return

    # 此回调在每轮验证结束后执行，metrics 已包含 args.nms 所选分支的精度。
    primary = trainer.metrics.get(BEST_METRIC_NAME)
    if primary is None:
        LOGGER.warning(f"未找到 {BEST_METRIC_NAME}，本轮无法记录最佳 OBB 精度。")
        return

    # args.nms is False -> 框架验证的是 one2one，附加评估得到的就是 one2many，反之亦然。
    primary_branch = BRANCH_ONE2ONE if trainer.args.nms is False else BRANCH_ONE2MANY
    other_branch = BRANCH_ONE2MANY if primary_branch == BRANCH_ONE2ONE else BRANCH_ONE2ONE
    scores = {primary_branch: float(primary)}
    other = evaluate_other_branch(trainer)
    if other is not None:
        scores[other_branch] = float(other)
        scores["mean"] = (scores[BRANCH_ONE2ONE] + scores[BRANCH_ONE2MANY]) / 2

    states = getattr(trainer, "obb_best_states", None)
    if states is None:
        states = {name: {"map": float("-inf"), "epoch": 0} for name in TRACK_TARGETS}
        trainer.obb_best_states = states

    # 本轮创下新高的追踪目标共用一份 last.pt 快照，避免重复导出同一个文件。
    improved = [name for name in TRACK_TARGETS if name in scores and scores[name] > states[name]["map"]]
    snapshot = None
    for name in improved:
        states[name] = {"map": scores[name], "epoch": epoch}
        if snapshot is None:
            # 此时 save_model() 已在当前验证轮写入 last.pt；导出为带轮次和精度的不可覆盖副本。
            tags = "_".join(f"{n}{scores[n]:.4f}" for n in improved)
            snapshot = trainer.wdir / f"best_epoch{epoch:03d}_{tags}.pt"
            # strip_optimizer 读 last.pt 另存为 snapshot，不改动 last.pt 本身 (续训仍可用)。
            # 它会丢掉 optimizer/scaler/stopper 等训练态、用 EMA 覆盖 model 并转 FP16，
            # 体积约为原来的 1/3。推理和导出只需要权重，所以信息没有损失。
            strip_optimizer(trainer.last, snapshot)
        states[name]["path"] = snapshot.name

    current = "  ".join(f"{name}={scores[name]:.4f}" for name in TRACK_TARGETS if name in scores)
    LOGGER.info(f"\nEpoch {epoch} mAP50-95: {current}")
    if improved:
        LOGGER.info(f"刷新最佳: {', '.join(improved)} -> {snapshot.name}")
    for name in TRACK_TARGETS:
        state = states[name]
        if state["epoch"]:
            LOGGER.info(f"  最佳 {name:9s}: Epoch {state['epoch']:3d} mAP50-95={state['map']:.4f} ({state['path']})")


def configure_validation_schedule(trainer) -> None:
    """在训练启动后关闭默认逐轮验证，并使用回调在固定间隔恢复验证。"""
    trainer.args.val = False
    LOGGER.info(f"验证策略: 每 {VALIDATE_EVERY} 轮执行一次验证，并按 mAP50-95 保存最佳 OBB 模型。")


def schedule_next_validation(trainer) -> None:
    """在每个训练轮开始前决定这一轮训练结束后是否需要验证。"""
    next_epoch = trainer.epoch + 1
    trainer.args.val = next_epoch % VALIDATE_EVERY == 0 or next_epoch >= trainer.epochs

# ======================================================================================================================
# 基础配置
# ======================================================================================================================
# 预训练模型 (premodel 目录下): yolo26n-obb.pt (更快) / yolo26s-obb.pt (更准)
MODEL = "premodel/yolo26n-obb.pt"
# 如需从零训练可改用模型结构文件: "ultralytics/cfg/models/26/yolo26-obb.yaml"
# 数据集配置文件, 替换为你自己的 OBB 数据集 yaml; dota8.yaml 为官方示例 (首次运行会自动下载)
DATA = "ultralytics/cfg/datasets/dota8.yaml"

# ======================================================================================================================
# 训练参数 (与官方 train 模式参数保持一致)
# ======================================================================================================================
TRAIN_ARGS = dict(
    # ------------------------------------------------- 训练设置 -------------------------------------------------
    data=DATA,  # (str) 数据集 yaml 路径
    task="obb",  # (str) 任务类型: 旋转框检测
    epochs=200,  # (int) 训练轮数
    time=None,  # (float) 最大训练小时数, 设置后会覆盖 epochs
    patience=0,  # (int) 0 表示关闭早停，避免未验证轮复用旧精度而触发错误的提前停止
    batch=4,  # (int|float) 批大小; -1 自动批大小; 0~1 小数表示按 GPU 显存占比自动设置
    imgsz=800,  # (int) 输入图像尺寸 (遥感大图可调大至 1024)
    save=True,  # (bool) 保存训练权重和结果
    save_period=-1,  # (int) 每 N 轮保存一次权重, <1 表示禁用
    cache=False,  # (bool|str) 缓存图片: True/'ram' 内存, 'disk' 磁盘, False 不缓存
    device=None,  # (int|str|list) 训练设备: 0 / [0,1] / 'cpu' / 'mps'; None 自动选择
    workers=8,  # (int) 数据加载线程数 (Windows 下如报错可调小或设为 0)
    project="runs/obb",  # (str) 结果保存根目录
    name="train",  # (str) 实验名称, 结果保存在 project/name
    exist_ok=False,  # (bool) 是否覆盖已存在的同名实验目录
    pretrained=True,  # (bool|str) 是否使用预训练权重, 或指定权重路径
    cls_remap=True,  # (bool) 跨数据集微调时按类别名重映射预训练分类头
    optimizer="auto",  # (str) 优化器: SGD, MuSGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp, auto
    verbose=True,  # (bool) 打印详细日志
    seed=0,  # (int) 随机种子
    deterministic=True,  # (bool) 启用确定性算法 (可复现, 但可能变慢)
    single_cls=False,  # (bool) 将所有类别视为单一类别
    rect=False,  # (bool) 矩形训练 (与 mosaic 不兼容)
    cos_lr=True,  # (bool) 使用余弦学习率调度
    close_mosaic=10,  # (int) 最后 N 轮关闭 mosaic 增强 (0 表示不关闭)
    resume=False,  # (bool) 从上次中断的 last.pt 恢复训练
    amp=True,  # (bool|str) 混合精度训练: True/'fp16', 'bf16', False/'fp32'
    fraction=1.0,  # (float) 使用训练集的比例
    profile=False,  # (bool) 训练时分析 ONNX/TensorRT 速度
    freeze=None,  # (int|list) 冻结前 N 层, 或指定层索引列表
    multi_scale=0.0,  # (float) 多尺度训练范围 (imgsz 的比例), 0 表示关闭
    compile=False,  # (bool|str) 启用 torch.compile
    # ------------------------------------------------- 验证设置 -------------------------------------------------
    val=False,  # (bool) 由回调控制: 每 VALIDATE_EVERY 轮与最后一轮执行验证
    plots=False,  # (bool) 保存训练/验证曲线和可视化图片
    # (bool|None) 选择验证时使用哪个检测头，判定条件是 `nms is False`:
    #   None  -> one2many 分支 + 外部 NMS (框架默认)
    #   (不用)True  -> 同样是 one2many 分支，仅在 export 时把 NMS 烘进计算图
    #   False -> one2one (NMS-free / End2End) 分支
    # 两个分支的损失始终同时参与训练 (E2ELoss)，此处只影响验证精度与 best.pt 的筛选依据。
    # 若最终要导出 End2End ONNX，必须设为 False，否则 best.pt 是按 one2many 最优轮次挑的。
    nms=None,  # (bool|None) 按 one2one (End2End) 分支验证并筛选最佳权重
    # ------------------------------------------------- 学习率与优化器 -------------------------------------------------
    lr0=0.003,  # (float) 初始学习率 (SGD=1e-2, Adam/AdamW=1e-3); optimizer='auto' 时会被自动覆盖
    lrf=0.01,  # (float) 最终学习率系数, 最终 lr = lr0 * lrf
    momentum=0.937,  # (float) SGD 动量 / Adam beta1
    weight_decay=0.0005,  # (float) 权重衰减 (L2 正则)
    warmup_epochs=3.0,  # (float) 预热轮数
    warmup_momentum=0.8,  # (float) 预热初始动量
    warmup_bias_lr=0.1,  # (float) 预热阶段 bias 学习率
    nbs=64,  # (int) 名义批大小, 用于梯度累积和损失归一化
    # ------------------------------------------------- 损失权重 -------------------------------------------------
    box=7.5,  # (float) 边框损失权重
    cls=0.5,  # (float) 分类损失权重
    cls_pw=0.0,  # (float) 类别不平衡加权幂次 (0 关闭, 1 完全按逆频率)
    dfl=1.5,  # (float) 分布焦点损失权重
    angle=1.0,  # (float) 旋转角度损失权重 (OBB 专用)
    # ------------------------------------------------- 数据增强 -------------------------------------------------
    hsv_h=0.015,  # (float) 色调增强幅度
    hsv_s=0.7,  # (float) 饱和度增强幅度
    hsv_v=0.4,  # (float) 亮度增强幅度
    degrees=30.0,  # (float) 随机旋转角度 (+/-), OBB 任务可适当设置如 10~180
    translate=0.1,  # (float) 随机平移比例
    scale=0.5,  # (float) 随机缩放幅度
    shear=0.0,  # (float) 随机剪切角度
    perspective=0.0,  # (float) 透视变换系数 (0~0.001)
    flipud=0.0,  # (float) 上下翻转概率 (遥感/俯视图像可设为 0.5)
    fliplr=0.5,  # (float) 左右翻转概率
    bgr=0.3,  # (float) RGB/BGR 通道交换概率
    mosaic=0.0,  # (float) Mosaic 拼接增强概率
    mixup=0.0,  # (float) MixUp 增强概率
    cutmix=0.0,  # (float) CutMix 增强概率
    copy_paste=0.0,  # (float) 复制粘贴增强比例 (支持 OBB)
    copy_paste_mode="flip",  # (str) 复制粘贴策略: flip / mixup
)


def main():
    # 修复源码版 Ultralytics 在未安装 polars 时，保存检查点会中断的问题。
    patch_results_csv_reader()

    # 加载预训练模型，并注册“每 5 轮验证和保存最佳模型”的回调。
    model = YOLO(MODEL)
    model.add_callback("on_train_start", configure_validation_schedule)
    model.add_callback("on_train_epoch_start", schedule_next_validation)
    model.add_callback("on_fit_epoch_end", validate_and_track_best)

    # 开始训练。框架仍会保存 last.pt 与 best.pt；最佳轮次副本位于 weights/best_epochXXX_*.pt。
    model.train(**TRAIN_ARGS)

    # 输出训练期间记录到的最佳结果；若训练未执行验证，则最后补充验证一次。
    trainer = model.trainer
    states = getattr(trainer, "obb_best_states", None)
    if not states or not any(state["epoch"] for state in states.values()):
        metrics = model.val(nms=TRAIN_ARGS.get("nms"))
        print(f"\n训练完成。训练期间未记录到验证结果，补充验证 mAP50-95: {metrics.box.map:.4f}")
    else:
        print("\n训练完成。各追踪目标的最佳权重:")
        for name in TRACK_TARGETS:
            state = states[name]
            if state["epoch"]:
                print(f"  {name:9s}: Epoch {state['epoch']:3d} mAP50-95={state['map']:.4f} -> {state['path']}")
    print(f"训练结果保存目录: {trainer.save_dir}")


if __name__ == "__main__":
    # Windows 下多进程数据加载需要放在 main 保护中
    main()
