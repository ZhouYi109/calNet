from dataclasses import dataclass
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class SyntheticConfig:
    num_sequences: int = 1000
    seq_len: int = 200
    window_size: int = 10
    noise_std: float = 0.02
    # 随机误差项范围（向量尺度、偏置）
    k_range: float = 0.1
    b_range: float = 0.2


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


def _random_walk_velocity(T: int, dt: float = 0.1) -> np.ndarray:
    v = np.zeros((T, 3), dtype=np.float32)
    a = np.random.randn(T, 3).astype(np.float32) * 0.1
    for t in range(1, T):
        v[t] = v[t - 1] + a[t] * dt
    return v


def make_synthetic_dataset(cfg: SyntheticConfig, em: str, split: str = 'train') -> WindowedDvlInsgnssDataset:
    rs = np.random.RandomState(42 if split == 'train' else 24)
    T = cfg.seq_len
    all_v_dvl = []
    all_v_insgnss = []
    for _ in range(cfg.num_sequences):
        v_gt = _random_walk_velocity(T)
        # INS-GNSS组合导航速度 = v_gt + 小噪声（已提供为b系速度）
        v_insgnss = v_gt + rs.randn(T, 3).astype(np.float32) * cfg.noise_std
        # 生成误差项（EM5全量，再由其他EM退化）
        k_vec = (rs.rand(3).astype(np.float32) - 0.5) * 2 * cfg.k_range
        b_vec = (rs.rand(3).astype(np.float32) - 0.5) * 2 * cfg.b_range
        if em == 'EM1':
            k_vec[:] = k_vec.mean()
            b_vec[:] = 0.0
        elif em == 'EM2':
            b_vec[:] = 0.0
        elif em == 'EM3':
            k_vec[:] = 0.0
            b_vec[:] = b_vec.mean()
        elif em == 'EM4':
            k_vec[:] = 0.0
        # 生成DVL观测： (1+k)*v + b + 噪声
        v_dvl = (1.0 + k_vec[None, :]) * v_gt + b_vec[None, :] + rs.randn(T, 3).astype(np.float32) * cfg.noise_std
        all_v_dvl.append(v_dvl)
        all_v_insgnss.append(v_insgnss)

    v_dvl = np.concatenate(all_v_dvl, axis=0)
    v_insgnss = np.concatenate(all_v_insgnss, axis=0)
    return WindowedDvlInsgnssDataset(v_dvl, v_insgnss, cfg.window_size)



