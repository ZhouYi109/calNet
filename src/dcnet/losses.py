import torch
import torch.nn as nn
import math

def rotation_matrix_from_euler_errors(pitch_error: torch.Tensor, yaw_error: torch.Tensor, roll_error: torch.Tensor) -> torch.Tensor:
    """
    根据俯仰角误差、偏航角误差和横滚角误差构建旋转矩阵C_b^d(φ)
    
    Args:
        pitch_error: 俯仰角误差 (B, 1)
        yaw_error: 偏航角误差 (B, 1) 
        roll_error: 横滚角误差 (B, 1)
    
    Returns:
        rotation_matrix: 旋转矩阵 (B, 3, 3)
    """
    B = pitch_error.shape[0]
    device = pitch_error.device
    
    # 提取角度
    pitch = pitch_error.squeeze(-1)  # (B,)
    yaw = yaw_error.squeeze(-1)      # (B,)
    roll = roll_error.squeeze(-1)    # (B,)
    
    # 计算三角函数值
    cp = torch.cos(pitch)
    sp = torch.sin(pitch)
    cy = torch.cos(yaw)
    sy = torch.sin(yaw)
    cr = torch.cos(roll)
    sr = torch.sin(roll)
    
    # 构建旋转矩阵 (ZYX顺序: 先绕Z轴(yaw)，再绕Y轴(pitch)，最后绕X轴(roll))
    # R = Rz(yaw) * Ry(pitch) * Rx(roll)
    
    # Rx(roll) - 绕X轴旋转
    Rx = torch.zeros(B, 3, 3, device=device)
    Rx[:, 0, 0] = 1
    Rx[:, 1, 1] = cr
    Rx[:, 1, 2] = -sr
    Rx[:, 2, 1] = sr
    Rx[:, 2, 2] = cr
    
    # Ry(pitch) - 绕Y轴旋转
    Ry = torch.zeros(B, 3, 3, device=device)
    Ry[:, 0, 0] = cp
    Ry[:, 0, 2] = sp
    Ry[:, 1, 1] = 1
    Ry[:, 2, 0] = -sp
    Ry[:, 2, 2] = cp
    
    # Rz(yaw) - 绕Z轴旋转
    Rz = torch.zeros(B, 3, 3, device=device)
    Rz[:, 0, 0] = cy
    Rz[:, 0, 1] = -sy
    Rz[:, 1, 0] = sy
    Rz[:, 1, 1] = cy
    Rz[:, 2, 2] = 1
    
    # 组合旋转矩阵: R = Rz * Ry * Rx
    R = torch.bmm(torch.bmm(Rz, Ry), Rx)
    return R

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
    if em == 'EM':
        # EM6: 包含比例因子和旋转矩阵的校准
        # params: (B, 6) - [scale_factor_x, scale_factor_y, scale_factor_z, pitch_error, yaw_error, roll_error]
        scale_factors = params[:, :3].view(B, 3, 1)  # δk: (B, 3, 1) 三轴分别的比例因子
        pitch_error = params[:, 3:4]  # 俯仰角误差
        yaw_error = params[:, 4:5]    # 偏航角误差  
        roll_error = params[:, 5:6]   # 横滚角误差
        
        # 构建旋转矩阵 C_b^d(φ)
        rotation_matrix = rotation_matrix_from_euler_errors(pitch_error, yaw_error, roll_error)
        
        # 应用校准: (1 + δk) * C_b^d(φ) * v_dvl
        # 注意：这里假设v_dvl已经是正确的坐标系，如果需要转换坐标系，可能需要调整
        v_rotated = torch.bmm(rotation_matrix, v_dvl)  # (B, 3, K)
        v_cal = (1.0 + scale_factors) * v_rotated
        return v_cal
    raise ValueError(f'Unknown EM: {em}')




class CalibrationMSELoss(nn.Module):
    """根据论文式(24)，最小化校准后速度与参考速度(INS-GNSS)的MSE。"""

    def __init__(self, em: str):
        super().__init__()
        self.em = em
        self.mse = nn.MSELoss()

    def forward(self, params: torch.Tensor, v_dvl: torch.Tensor, v_insgnss: torch.Tensor) -> torch.Tensor:
        v_cal = apply_em_correction(self.em, params, v_dvl)
        return self.mse(v_cal, v_insgnss)



