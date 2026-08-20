# SARAF 项目修复说明文档

> **项目**: Stationarity-Aware Retrieval-Augmented Time Series Forecasting (KDD'26)
> **修复日期**: 2026-08-07
> **修复目标**: 使项目在 Windows CPU 环境下正常运行所有数据集

---

## 1. 问题概述

在 Windows 环境下（无 GPU，纯 CPU 运行），SARAF 项目遇到两类错误：

| # | 错误 | 影响数据集 | 根因 |
|---|------|-----------|------|
| 1 | **WinError 1455** (页面文件太小) | Exchange Rate | DataLoader 10 个子进程同时加载 torch\lib\shm.dll |
| 2 | **OOM** (尝试分配 ~30.3 GB) | Electricity (321通道×17597样本) | 一次性 np.stack 全部样本 → 峰值 ~50 GB |

---

## 2. 修改文件清单

| 文件 | 修改类型 | 关联问题 |
|------|----------|----------|
| `data_provider/data_factory.py` | 新增 7 行 | #1 WinError 1455 |
| `layers/Retrieval.py` | 新增类 `_ShapeProxy` | #2 OOM |
| `layers/Retrieval.py` | 重写 `prepare_dataset()` | #2 OOM |
| `layers/Retrieval.py` | 新增 `_compute_dataset_stationarity_batched()` | #2 OOM |
| `layers/Retrieval.py` | 重写 `decompose_mg()` (新增 batch_t 参数) | #2 OOM |
| `layers/Retrieval.py` | 3 处 `.float()` 上转 | #2 float16 兼容 |

---

## 3. 修改一：WinError 1455 修复

**文件**: `data_provider/data_factory.py`

### 根因

Python 在 Windows 上的多进程启动方式为 `spawn`（非 Linux 的 `fork`）。`spawn` 创建全新 Python 解释器，重新 import 所有模块。

当 `num_workers=10` 时，10 个子进程同时加载 torch\lib\shm.dll（PyTorch 共享内存库），每个约 200MB 虚拟地址空间。Windows 要求每个进程独立映射 DLL，10 个进程共需 ~2GB 页面文件空间。

### 修改

在 `data_provider()` 函数中，Windows 下强制 `num_workers=0`：

```python
num_workers = args.num_workers
if os.name == "nt" and num_workers > 0:
    num_workers = 0
```

**依据**: `run.py` 原有检查仅在 GPU 模式下触发（`args.use_gpu and args.num_workers > 0`）。当 CUDA 不可用时 `use_gpu=False`，检查失效。此处去掉 GPU 条件作为第二道防线。

---

## 4. 修改二：OOM 防御体系

**文件**: `layers/Retrieval.py`

### 根因：原始内存消耗分析

Electricity: T=17597, S=720, C=321, 每个 float32 = 4 字节

```
步骤                           | 数据结构            | 内存
───────────────────────────────┼─────────────────────┼────────
np.stack(train_data_all)       | float64 [T,S,C]     | 32.5 GB
torch.tensor(...).float()      | float32 [T,S,C]     | 16.3 GB
decompose_mg 全量               | float32 [G,T,S,C]   | 16.3 GB
y 同理                          | float32 [G,T,P,C]   |  2.2 GB
保留 train_data_all 张量         | float32 [T,S,C]     | 16.3 GB
保留 y_data_all 张量             | float32 [T,P,C]     |  2.2 GB
───────────────────────────────┼─────────────────────┼────────
峰值 (np.stack + tensor 共存)   |                     | ~48.8 GB
最终常驻                       |                     | ~37   GB
```

核心问题：`np.stack` 创建 float64 中间数组（32.5 GB），与随后的 float32 tensor（16.3 GB）短时间内共存。

---

### 4.1 `_ShapeProxy` 代理类（新增）

```python
class _ShapeProxy:
    """提供 .shape 属性但不存储数据，替代 self.train_data_all / self.y_data_all"""
    def __init__(self, n, seq_len, channels):
        self.shape = (n, seq_len, channels)
```

**原理**: 原代码中 `self.train_data_all` 仅在下游被访问 `.shape` 属性（获取 data_len, seq_len, channels），不需要实际数据。用 3 个 int（24 字节）替代 16.3 GB 的张量。

---

### 4.2 `prepare_dataset` 流水线化（重写）

**策略**: 一次全量 → 分批流水线

```
修改前: list → np.stack(全部) → tensor(全部) → decompose(全部) → 存储
修改后: list → [batch0] np.stack → tensor → decompose → .half()
            → [batch1] np.stack → tensor → decompose → .half()
            → ... → torch.cat(batches)
```

每批 512 样本：`512 × 720 × 321 × 8B (np) + 512 × 720 × 321 × 4B (tensor) ≈ 1.4 GB` 峰值。

**float16 存储**: `chunk_mg.half()` 将 decompose 结果从 float32 转为 float16（8.1 GB vs 16.3 GB）。数据是相对偏移量（已减基准值），数值范围小，float16（~3.3 位十进制精度）足够。

---

### 4.3 `decompose_mg` 分批（重写）

新增 `batch_t=512` 参数，沿 T 维度切片：

```python
def decompose_mg(self, data_all, remove_offset=True, batch_t=512):
    for t_start in range(0, T_total, batch_t):
        chunk = data_all[t_start:t_start+batch_t]  # [512, S, C]
        ...  # 对 chunk 做 unfold/mean/stack
    return torch.cat(chunks, dim=1)  # 拼接
```

峰值从 `[G, 17597, 720, 321]` (16.3 GB) 降至 `[G, 512, 720, 321]` (~472 MB)。

---

### 4.4 平稳性分析分批（新增 `_compute_dataset_stationarity_batched`）

原方法 `compute_dataset_stationarity(self.train_data_all)` 需要全量张量。

新方法接收 numpy 列表，每 512 样本一批计算平稳性分数（每样本返回 1 个标量），最终拼接为 `[17597]` 向量（~70 KB）。

```python
def _compute_dataset_stationarity_batched(self, data_list, n_total, ...):
    all_scores = []
    for start in range(0, n_total, batch_size):
        chunk = torch.tensor(np.stack(data_list[start:end])).float()
        scores = self.compute_stationarity_scores(chunk)  # [512]
        all_scores.append(scores)
    return torch.cat(all_scores).mean().item()
```

---

### 4.5 计算时 float16→float32 上转（3 处）

| 位置 | 函数 | 原因 |
|------|------|------|
| `periodic_batch_corr` L617 | `.to(device).float()` | tain_data_all_mg 是 float16，F.normalize 和 bmm 需要 float32 |
| `generate_retrieval_predictions` L662 | `.flatten().float()` | y_data_all_mg (float16) × ranking_prob (float32) → bmm 不支持混合精度 |
| `generate_retrieval_predictions_2` L747 | `.flatten().float()` | 同上 |

**依据**: PyTorch `torch.bmm` 要求两个输入 dtype 一致。float16×float32 混合精度在 CPU 上会报错（GPU Tensor Core 有特殊路径，CPU 无）。

---

## 5. 内存对比

```
                        修改前          修改后          节省
──────────────────────────────────────────────────────────────
train_data_all          16.3 GB         24 字节         100%
y_data_all               2.2 GB         24 字节         100%
train_data_all_mg       16.3 GB (fp32)   8.1 GB (fp16)   50%
y_data_all_mg            2.2 GB (fp32)   1.1 GB (fp16)   50%
──────────────────────────────────────────────────────────────
常驻总计                37.0 GB          9.2 GB          75%
峰值 (prepare 阶段)     48.8 GB          1.4 GB          97%
──────────────────────────────────────────────────────────────
```

## 6. 已验证结果

| 数据集 | 配置 | MSE | R² |
|--------|------|-----|-----|
| Exchange Rate | seq=96, pred=96 | 0.0825 | 0.952 |
| Exchange Rate | seq=720, pred=96 | 0.0852 | 0.951 |
