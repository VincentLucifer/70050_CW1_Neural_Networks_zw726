# Neural Networks — IC Machine Learning Coursework

[简体中文](README.md) | [English](README_EN.md)

Imperial College London（IC）Machine Learning 课程的神经网络作业：使用 NumPy 实现模块化神经网络与手动反向传播，再使用 PyTorch 构建 California Housing 房价回归流程。

本项目基于学校提供的 starter code、接口与数据完成课程实现，用于展示机器学习基础与工程实践，并非所有代码均为独立原创。原始作业说明完整保留在 [Coursework brief](docs/COURSEWORK_ORIGINAL.md)。

## 核心实现

### Part 1：NumPy 神经网络

- 实现 `LinearLayer`、`SigmoidLayer`、`ReluLayer` 的前向计算与反向传播，缓存中间值以计算梯度。
- `MultiLayerNetwork` 支持组合网络及 `sigmoid`、`relu`、`identity` 激活，配合 Xavier 初始化、MSE / softmax cross-entropy 损失和 mini-batch SGD。
- `Preprocessor` 提供可逆 min-max scaling，并处理常量特征。
- Iris 示例采用 4 → 16 → 3 网络，按 80/20 划分训练与验证数据，打印 loss 与分类 accuracy。

### Part 2：PyTorch 房价回归

- `Regressor` 提供 `fit`、`predict`、`evaluate` 接口，默认隐藏层为 128 → 64 → 32，使用 ReLU 与线性输出层。
- 浮点列缺失值中位数填充、min-max scaling、类别字段编码，以及目标尺度还原。
- 特征工程包含每户房间数、每户人口、卧室比例、`log1p` 特征和到六个加州城市的经纬度欧氏距离。这些距离是坐标空间近似，并非公里数。
- Adam、weight decay、cosine learning-rate scheduling 和 mini-batch 训练；使用原始目标尺度上的 RMSE 评估。
- 入口先在 80/20 划分上训练验证模型，再用全量数据训练最终模型并保存，区分验证误差与最终模型训练误差。

## 代码导航

| 文件 | 职责 |
| --- | --- |
| [part1_nn_lib.py](part1_nn_lib.py) | NumPy 网络层、损失、训练器、预处理与 Iris 示例 |
| [part2_house_value_regression.py](part2_house_value_regression.py) | PyTorch 回归器、特征工程、训练与序列化 |
| `iris.dat` / `housing.csv` | 随课程仓库提供的 Iris / California Housing 数据 |
| `part2_model.pickle` | 已纳入版本控制的模型；本次未加载或评估 |
| [requirements.txt](requirements.txt) | 固定版本的 Python 依赖 |
| [docs/COURSEWORK_ORIGINAL.md](docs/COURSEWORK_ORIGINAL.md) | 原始作业描述与提交说明 |

技术栈：Python、NumPy（数组与手动梯度）、PyTorch（自动微分与优化）、pandas（表格处理）、scikit-learn（类别编码）。完整依赖见 `requirements.txt`。

## 安装与运行

需要 Python 与 pip，从仓库根目录执行：

```powershell
python -m pip install -r requirements.txt

# Iris 分类示例：输出 loss / accuracy
python part1_nn_lib.py

# 验证集评估，然后全量训练并保存模型
python part2_house_value_regression.py
```

Part 1 示例训练 1,000 epochs；Part 2 入口分别训练两个 200-epoch 模型。运行 Part 2 会覆盖当前目录的 `part2_model.pickle`。

命令根据源码入口与依赖文件整理。本次文档更新没有安装依赖、执行训练或运行测试；依赖版本可用性与环境兼容性尚未验证。

## 技术讨论与当前边界

面试讨论可从 `LinearLayer.backward` 的矩阵梯度、网络逆序传播、训练预处理参数的复用，以及验证模型与最终提交模型的区别展开。

- 本次未复核模型指标或 LabTS 结果，不列出未经验证的性能数值或课程成绩。
- 当前采用单次随机 holdout，未提供交叉验证、基线对比或独立实验报告。
- Part 2 预处理主要面向提供的 housing 数据；常量数值列、二分类文本列和未见类别等输入边界仍需完善。
- Part 2 在 `fit` 中设置种子，但网络通常已在构造函数中初始化，因此不足以保证端到端可复现。
- 后续重点：固定完整随机状态、记录可复现的基线与验证结果、完善预处理边界。

## 课程来源与原始要求

**Imperial College London — Machine Learning，Neural Networks coursework。** 学校提供原始任务框架与数据；本仓库保留课程接口及说明，不代表学校官方发布。

原作业要求通过 LabTS 测试、提交最终 commit，并将最终 `part2_model.pickle` 纳入仓库。GitHub 副本用于个人整理和项目展示，课程提交仍按学校流程进行。[完整原始作业说明](docs/COURSEWORK_ORIGINAL.md) 包含原环境设置与提交步骤。
