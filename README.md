DCNet 参考实现（基于论文：DCNet: A Data-Driven Framework for DVL Calibration, 2024）

本仓库提供一个可运行的参考实现，覆盖：
- 2D-CNN 双分支（IDCNN 与 2DCNN 头）+ 全连接头（FC Head）
- 五类DVL误差模型 EM1-EM5（尺度因子/偏置的标量或向量组合）
- 训练流程（MSE损失）、窗口化输入、合成数据演示
- **所有参数已按照论文Table 1设置**

## 安装
```bash
pip install -r requirements.txt
```

## 快速开始

### Python版本（推荐）
```bash
# 使用论文默认参数运行（epochs=100, batch_size=256）
python -m src.dcnet.train

# 自定义参数
python -m src.dcnet.train --em EM5 --epochs 10 --batch_size 128 --window_size 10
```

### MATLAB版本
```matlab
cd matlab
run_example
```

## 目录结构
```
src/dcnet/
  model.py      # DCNet模型定义
  losses.py     # MSE损失函数
  data.py       # 数据生成和加载
  train.py      # 训练主程序
matlab/
  *.m           # MATLAB实现版本
```

## 论文参数设置（Table 1）

所有超参数已按照论文Table 1设置：

| 参数 | 值 | 说明 |
|------|-----|------|
| **Epochs** | 100 | 训练轮数 |
| **Batch Size** | 256 | 批量大小 |
| **LeakyReLU** | 0.05 | 负斜率 |
| **Dropout** | 0.3 | Dropout概率 |
| **FC结构** | 128→256→128→out | 全连接层 |
| **窗口大小** | 10 | 时间窗口K |

## 网络架构

- **1DCNN Head**: 输入通道3，隐藏层32通道，2层卷积
- **2DCNN Head**: 输入通道6，隐藏层16通道，3层卷积
- **FC Head**: 4层全连接（128→256→128→out）

## 数据接口要点（接入真实数据时）

- 输入 DVL 与 GNSS-RTK 速度（均在DVL机体系）
- IDCNN 分支使用差分 `v_dvl - v_gnss`；2DCNN 分支拼接两者得到通道数6
- 按 EM 选择输出维度：EM1/EM3 标量，EM2/EM4 向量，EM5 同时估计两者

## 注意事项

该实现为参考工程骨架，真实复现实验需替换真实数据与评估流程。