from dataclasses import dataclass
import numpy as np
import torch
import os
from torch.utils.data import Dataset

class WindowedDvlInsgnssDataset(Dataset):
    def __init__(self, v_dvl: np.ndarray, v_insgnss: np.ndarray, window_size: int):
        assert v_dvl.shape == v_insgnss.shape
        assert v_dvl.shape[1] == 3
        self.v_dvl = v_dvl.astype(np.float32)
        self.v_insgnss = v_insgnss.astype(np.float32)
        self.K = window_size
        self.N = v_dvl.shape[0]
        self.max_start = self.N - self.K

    def __len__(self):
        return self.max_start

    def __getitem__(self, idx: int):
        v_d = self.v_dvl[idx:idx + self.K].T  # (3,K)
        v_i = self.v_insgnss[idx:idx + self.K].T
        return torch.from_numpy(v_d), torch.from_numpy(v_i)

def make_csv_dataset(csv_path: str = 'vb_vl_t.csv', window_size: int = 10, split: str = 'train') -> WindowedDvlInsgnssDataset:
    """
    从 CSV 读取速度与时间数据，并返回与 data.py 中完全一致的 WindowedDvlInsgnssDataset。

    约定：
    - 前三列为 vb (v_insgnss)
    - 第四到第六列为 v_dvl
    - 最后一列为时间序列（此处读取但不参与数据集拼接）

    参数：
    - csv_path: CSV 文件路径，默认项目根目录下 'vb_vl_t.csv'
    - window_size: 窗口大小 K
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f'CSV 文件未找到: {csv_path}')

    # 为兼容不同分隔符（单/多空格、制表符、逗号），使用 genfromtxt 自动解析并去除空白
    # 同时过滤含 NaN 的行以避免后续处理失败
    data = np.genfromtxt(
        csv_path,
        delimiter=None,       # 自动将任意空白/逗号视为分隔符
        dtype=np.float32,
        autostrip=True,       # 去除前后空白
        comments=None         # 不将任何字符视为注释前缀
    )
    if data.ndim == 1:
        data = data.reshape(1, -1)
    # 丢弃包含 NaN 的行（例如由杂质文本或空列引起）
    if np.isnan(data).any():
        data = data[~np.isnan(data).any(axis=1)]
    if data.ndim != 2 or data.shape[1] < 7:
        raise ValueError(
            f'CSV 维度不符合预期，获得形状 {data.shape}，应至少包含7列(3+3+time)'
        )
    if split == 'train':
        data = data[:600000]
    elif split == 'val':
        data = data[600000:]
    # 列选择：0~2 -> v_insgnss，3~5 -> v_dvl，-1 -> time（不使用）
    v_insgnss = data[:, 0:3]
    v_dvl = data[:, 3:6]
    # time = data[:, -1]  # 如果后续需要可返回或用于对齐

    # 与 data.py 保持一致：返回 WindowedDvlInsgnssDataset(v_dvl, v_insgnss, window_size)
    return WindowedDvlInsgnssDataset(v_dvl=v_dvl, v_insgnss=v_insgnss, window_size=window_size)


