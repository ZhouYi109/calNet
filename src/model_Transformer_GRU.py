import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (seq_len, batch, d_model)
        return x + self.pe[:x.size(0), :]


class TransformerEncoder(nn.Module):
    """Transformer encoder for temporal feature extraction."""
    
    def __init__(self, d_model: int = 128, nhead: int = 8, num_layers: int = 2, p_drop: float = 0.3):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        
        # Input projection
        self.input_proj = nn.Linear(9, d_model)  # 9 channels: v_dvl(3) + v_insgnss(3) + v_sub(3)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=p_drop,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Layer normalization and dropout
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, K, 9) - already transposed from (B, 9, K)
        B, K, _ = x.shape
        
        # Input projection
        x = self.input_proj(x)  # (B, K, d_model)
        
        # Add positional encoding (transpose for positional encoding, then transpose back)
        x = x.transpose(0, 1)  # (K, B, d_model)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # (B, K, d_model)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Transformer encoding
        x = self.transformer(x)  # (B, K, d_model)
        
        # Layer normalization
        x = self.norm(x)
        
        return x


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


class Transformer_GRU(nn.Module):
    """Transformer+GRU architecture compatible with existing training/evaluation code.

    Inputs:
      - v_dvl, v_insgnss: (B, 3, K)
    Output:
      - calibration parameters shaped according to `em`
    """

    def __init__(self, window_size: int, em: str = 'EM5', p_drop: float = 0.3):
        super().__init__()
        self.em = em.upper()

        # Transformer encoder for feature extraction
        transformer_dim = 128
        self.transformer = TransformerEncoder(
            d_model=transformer_dim,
            nhead=8,
            num_layers=4,
            p_drop=p_drop
        )

        # GRU for temporal modeling
        gru_hidden = 128
        self.gru = nn.GRU(
            input_size=transformer_dim,
            hidden_size=gru_hidden,
            num_layers=4,
            batch_first=True,
            dropout=p_drop,
            bidirectional=True,
        )

        # Fully-connected head from GRU pooled features to EM parameters
        gru_out_dim = gru_hidden * 2  # bidirectional
        self.fc = FCHead(in_features=gru_out_dim, out_dim=output_dim_for_em(self.em), p_drop=p_drop)

    def forward(self, v_dvl: torch.Tensor, v_insgnss: torch.Tensor) -> torch.Tensor:
        # v_dvl, v_insgnss: (B, 3, K)
        v_sub = v_dvl - v_insgnss  # (B, 3, K)
        x = torch.cat([v_dvl, v_insgnss, v_sub], dim=1)  # (B, 9, K)

        # Transpose for transformer: (B, K, 9)
        x = x.transpose(1, 2)

        # Transformer feature extraction
        transformer_out = self.transformer(x)  # (B, K, d_model)

        # GRU encoding
        # out: (B, K, 2*H), hn: (num_layers*2, B, H)
        gru_out, hn = self.gru(transformer_out)

        # Use the last time-step features (or mean-pool); here we concat last and mean for robustness
        last_feat = gru_out[:, -1, :]  # (B, 2*H)
        mean_feat = torch.mean(gru_out, dim=1)  # (B, 2*H)
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
