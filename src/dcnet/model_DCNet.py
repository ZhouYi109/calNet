import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dilation: int = 1, p_drop: float = 0.3):
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.act = nn.LeakyReLU(0.05, inplace=True)
        self.drop = nn.Dropout(p_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.act(x)
        x = self.drop(x)
        return x


class ConvBlock2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dilation: int = 1, p_drop: float = 0.3):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.act = nn.LeakyReLU(0.05, inplace=True)
        self.drop = nn.Dropout(p_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.act(x)
        x = self.drop(x)
        return x


class IDCNNHead(nn.Module):
    """1D-CNN Head using subtraction input (v_dvl - v_insgnss)."""

    def __init__(self, in_channels: int = 3, hidden_channels: int = 32, p_drop: float = 0.3):
        super().__init__()
        # 两层1D卷积，论文描述第二层后接LeakyReLU；第一层膨胀率3x1，其他1x1
        self.c1 = ConvBlock1D(in_channels, hidden_channels, kernel_size=3, dilation=3, p_drop=p_drop)
        self.c2 = ConvBlock1D(hidden_channels, hidden_channels, kernel_size=3, dilation=1, p_drop=p_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, K)
        h = self.c1(x)
        h = self.c2(h)
        # 展平成 (B, C*K)
        return torch.flatten(h, start_dim=1)


class D2CNNHead(nn.Module):
    """2D-CNN Head processing concatenated velocities (DVL and INS-GNSS)."""

    def __init__(self, in_channels: int = 6, hidden_channels: int = 16, p_drop: float = 0.3):
        super().__init__()
        # 三层2D卷积，第一层膨胀3x1，其余1x1
        self.c1 = ConvBlock2D(in_channels, hidden_channels, kernel_size=3, dilation=1, p_drop=p_drop)
        self.c2 = ConvBlock2D(hidden_channels, hidden_channels, kernel_size=3, dilation=1, p_drop=p_drop)
        self.c3 = ConvBlock2D(hidden_channels, hidden_channels, kernel_size=3, dilation=1, p_drop=p_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 6, K, 1)
        h = self.c1(x)
        h = self.c2(h)
        h = self.c3(h)
        return torch.flatten(h, start_dim=1)


def output_dim_for_em(em: str) -> int:
    em = em.upper()
    if em == 'EM1':  # scalar scale factor k
        return 1
    if em == 'EM2':  # vector scale factor kx,ky,kz
        return 3
    if em == 'EM3':  # scalar bias b
        return 1
    if em == 'EM4':  # vector bias bx,by,bz
        return 3
    if em == 'EM5':  # both vectors (k_vec and b_vec)
        return 6
    raise ValueError(f'Unknown EM: {em}')


class FCHead(nn.Module):
    def __init__(self, in_features: int, out_dim: int, hidden: int = 128, p_drop: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.LeakyReLU(0.05, inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(hidden, 256),
            nn.LeakyReLU(0.05, inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(256, hidden),
            nn.LeakyReLU(0.05, inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DCNet(nn.Module):
    def __init__(self, window_size: int, em: str = 'EM5', p_drop: float = 0.3):
        super().__init__()
        self.em = em.upper()
        # Heads
        self.idcnn = IDCNNHead(in_channels=3, hidden_channels=32, p_drop=p_drop)
        self.d2cnn = D2CNNHead(in_channels=6, hidden_channels=16, p_drop=p_drop)

        # 估计flatten后的特征维度（用window_size推断）
        with torch.no_grad():
            dummy_id = torch.zeros(1, 3, window_size)
            fi = self.idcnn(dummy_id).shape[1]
            dummy_2d = torch.zeros(1, 6, window_size, 1)
            f2 = self.d2cnn(dummy_2d).shape[1]
        fused_dim = fi + f2

        self.fc = FCHead(in_features=fused_dim, out_dim=output_dim_for_em(self.em))

    def forward(self, v_dvl: torch.Tensor, v_insgnss: torch.Tensor) -> torch.Tensor:
        # v_dvl, v_insgnss: (B, 3, K)
        v_sub = v_dvl - v_insgnss
        h1 = self.idcnn(v_sub)
        x2 = torch.cat([v_dvl, v_insgnss], dim=1)  # (B, 6, K)
        x2 = x2.unsqueeze(-1)  # (B,6,K,1)
        h2 = self.d2cnn(x2)
        h = torch.cat([h1, h2], dim=1)
        out = self.fc(h)
        return out


def apply_em_correction(em: str, params: torch.Tensor, v_dvl: torch.Tensor) -> torch.Tensor:
    """根据EM输出将DVL速度校准，返回校准后的速度 v_cal。
    v_dvl: (B, 3, K)
    params: (B, D)
    """
    em = em.upper()
    B, _, K = v_dvl.shape
    if em == 'EM1':
        k = params.view(B, 1, 1)
        v_cal = (1.0 + k) * v_dvl
        return v_cal
    if em == 'EM2':
        k = params.view(B, 3, 1)
        v_cal = (1.0 + k) * v_dvl
        return v_cal
    if em == 'EM3':
        b = params.view(B, 1, 1)
        v_cal = v_dvl + b
        return v_cal
    if em == 'EM4':
        b = params.view(B, 3, 1)
        v_cal = v_dvl + b
        return v_cal
    if em == 'EM5':
        k = params[:, :3].view(B, 3, 1)
        b = params[:, 3:].view(B, 3, 1)
        v_cal = (1.0 + k) * v_dvl + b
        return v_cal
    raise ValueError(f'Unknown EM: {em}')



