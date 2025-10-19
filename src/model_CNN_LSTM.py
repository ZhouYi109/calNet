import torch
import torch.nn as nn
import torch.nn.functional as F


def output_dim_for_em(em: str) -> int:
    em = em.upper()
    if em == 'EM1':
        return 1
    if em == 'EM2':
        return 3
    if em == 'EM3':
        return 1
    if em == 'EM4':
        return 3
    if em == 'EM5':
        return 6
    raise ValueError(f'Unknown EM: {em}')


class ConvStack1D(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64, p_drop: float = 0.3):
        super().__init__()
        self.c1 = nn.Conv1d(in_channels, hidden_channels, kernel_size=5, padding=2)
        self.c2 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.act = nn.LeakyReLU(0.05, inplace=True)
        self.drop = nn.Dropout(p_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, K)
        x = self.c1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.c2(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.drop(x)
        return x  # (B, H, K)


class FCHead(nn.Module):
    def __init__(self, in_features: int, out_dim: int, hidden: int = 128, p_drop: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.LeakyReLU(0.05, inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.05, inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CNN_LSTM(nn.Module):
    """CNN+LSTM architecture compatible with existing training/evaluation code.

    Inputs:
      - v_dvl, v_insgnss: (B, 3, K)
    Output:
      - calibration parameters shaped according to `em`
    """

    def __init__(self, window_size: int, em: str = 'EM5', p_drop: float = 0.3):
        super().__init__()
        self.em = em.upper()

        # Use concatenation of DVL, INS-GNSS and their difference → 9 channels
        in_channels = 9
        conv_hidden = 64
        self.conv = ConvStack1D(in_channels=in_channels, hidden_channels=conv_hidden, p_drop=p_drop)

        # LSTM over the temporal dimension K. Input size equals conv hidden channels
        lstm_hidden = 128
        self.lstm = nn.LSTM(
            input_size=conv_hidden,
            hidden_size=lstm_hidden,
            num_layers=4,
            batch_first=True,
            dropout=p_drop,
            bidirectional=True,
        )

        # Fully-connected head from LSTM pooled features to EM parameters
        lstm_out_dim = lstm_hidden * 2  # bidirectional
        self.fc = FCHead(in_features=lstm_out_dim, out_dim=output_dim_for_em(self.em), p_drop=p_drop)

    def forward(self, v_dvl: torch.Tensor, v_insgnss: torch.Tensor) -> torch.Tensor:
        # v_dvl, v_insgnss: (B, 3, K)
        v_sub = v_dvl - v_insgnss  # (B, 3, K)
        x = torch.cat([v_dvl, v_insgnss, v_sub], dim=1)  # (B, 9, K)

        # CNN feature extraction along time
        h = self.conv(x)  # (B, Cc, K)

        # Prepare for LSTM: (B, K, Cc)
        h = h.transpose(1, 2)

        # LSTM encoding
        # out: (B, K, 2*H), (hn, cn): ((num_layers*2, B, H), (num_layers*2, B, H))
        out, (hn, cn) = self.lstm(h)

        # Use the last time-step features (or mean-pool); here we concat last and mean for robustness
        last_feat = out[:, -1, :]  # (B, 2*H)
        mean_feat = torch.mean(out, dim=1)  # (B, 2*H)
        fused = 0.5 * (last_feat + mean_feat)

        params = self.fc(fused)
        return params


def apply_em_correction(em: str, params: torch.Tensor, v_dvl: torch.Tensor) -> torch.Tensor:
    """Apply calibration parameters to DVL velocities according to EM.

    v_dvl: (B, 3, K)
    params: (B, D) where D depends on EM
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
