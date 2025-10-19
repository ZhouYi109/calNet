import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

from model_DCNet import DCNet
# from model_CNN_GRU import CNN_GRU
# from model_CNN_LSTM import CNN_LSTM
# from model_Transformer_GRU import Transformer_GRU
from losses import CalibrationMSELoss, apply_em_correction
from data import make_synthetic_dataset, SyntheticConfig
from data_csv import make_csv_dataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--em', type=str, default='EM5')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--window_size', type=int, default=10)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--dataset', type=str, default='csv')
    return p.parse_args()


def main():
    start_time = time.time()
    args = parse_args()
    device = torch.device(args.device)
    print('Using device:', device)
    if args.dataset == 'synthetic':
        cfg = SyntheticConfig(window_size=args.window_size)
        train_ds = make_synthetic_dataset(cfg, args.em, split='train')
        val_ds = make_synthetic_dataset(cfg, args.em, split='val')
    elif args.dataset == 'csv':
        train_ds = make_csv_dataset(window_size=args.window_size, split='train')
        val_ds = make_csv_dataset(window_size=args.window_size, split='val')
    else:
        raise NotImplementedError('无可用数据类型,请检查dataset参数值!')

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = DCNet(window_size=args.window_size, em=args.em).to(device)
    criterion = CalibrationMSELoss(em=args.em)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        model.train()
        train_loss_sum = 0.0
        for v_d, v_i in tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs + 1} train'):
        # for step,(v_d,v_i) in enumerate(train_loader):
            v_d = v_d.to(device)
            v_i = v_i.to(device)
            params = model(v_d, v_i)
            loss = criterion(params, v_d, v_i)
            optim.zero_grad()
            loss.backward()
            optim.step()
            train_loss_sum += loss.item() * v_d.size(0)
        train_loss = train_loss_sum / len(train_loader.dataset)

        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for v_d, v_i in tqdm(val_loader, desc=f'Epoch {epoch}/{args.epochs + 1} val'):
            # for step,(v_d,v_i) in enumerate(val_loader):
                v_d = v_d.to(device)
                v_i = v_i.to(device)
                params = model(v_d, v_i)
                loss = criterion(params, v_d, v_i)
                val_loss_sum += loss.item() * v_d.size(0)
        val_loss = val_loss_sum / len(val_loader.dataset)

        epoch_end_time = time.time()
        print(f'Epoch {epoch}: train_loss={train_loss:.6f} val_loss={val_loss:.6f}')
        print(f'Epoch {epoch} time: {epoch_end_time - epoch_start_time:.2f} seconds')
        # 记录曲线
        if epoch == 1:
            losses = {'train': [], 'val': []}
        losses['train'].append(train_loss)
        losses['val'].append(val_loss)

    # 绘图
    try:
        plt.figure(figsize=(5,3))
        plt.plot(losses['train'], label='train')
        plt.plot(losses['val'], label='val')
        plt.xlabel('epoch')
        plt.ylabel('MSE')
        plt.title(f'{model.__class__.__name__} {args.em} csv')
        plt.legend()
        out_path = 'training_curve_'+f'{model.__class__.__name__}_Ultra_csv'+'.png'
        plt.tight_layout()
        plt.savefig(out_path, dpi=160)
        print(f'Saved {out_path}')
    except Exception as e:
        print(f'Plot failed: {e}')

    # 评估：标定前后RMSE（按窗口-样本聚合）
    def batch_rmse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # a,b: (B,3,K)
        return torch.sqrt(torch.mean((a - b) ** 2, dim=(1,2)))

    base_all = []
    cal_all = []
    with torch.no_grad():
        for v_d, v_i in DataLoader(val_ds, batch_size=args.batch_size, shuffle=False):
            v_d = v_d.to(device)
            v_i = v_i.to(device)
            params = model(v_d, v_i)
            # 标定后速度
            v_cal = apply_em_correction(args.em, params, v_d)
            base_all.append(batch_rmse(v_d, v_i).cpu())
            cal_all.append(batch_rmse(v_cal, v_i).cpu())

    import torch as _torch
    base_rmse = _torch.cat(base_all).mean().item()
    cal_rmse = _torch.cat(cal_all).mean().item()
    imp = (base_rmse - cal_rmse) / max(base_rmse, 1e-12) * 100.0
    print(f'Validation RMSE (baseline vs calibrated): {base_rmse:.6f} -> {cal_rmse:.6f} ({imp:.2f}% improvement)')

    # 打印标定参数（取验证集前几个样本的平均）
    print(f'\n标定参数 ({args.em}):')
    with torch.no_grad():
        sample_params = []
        for i, (v_d, v_i) in enumerate(DataLoader(val_ds, batch_size=32, shuffle=False)):
            if i >= 3:  # 只看前3个batch
                break
            v_d = v_d.to(device)
            v_i = v_i.to(device)
            params = model(v_d, v_i)
            sample_params.append(params.cpu())
        
        avg_params = torch.cat(sample_params).mean(dim=0)
        
        # 获取模型名称
        model_name = model.__class__.__name__
        
        # 写入EM.txt文件
        # with open('EM.txt', 'w', encoding='utf-8') as f:
        # 写入EM.txt文件（追加，不覆盖）
        with open('EM.txt', 'a', encoding='utf-8') as f:
            f.write(f"\n模型名称: {model_name}--Ultra--csv\n")
            f.write(f"误差模型: {args.em}\n")
            f.write("=" * 50 + "\n")
            
            if args.em == 'EM1':
                print(f'  标量尺度因子 k: {avg_params[0]:.6f}')
                f.write(f"标量尺度因子 k: {avg_params[0]:.6f}\n")
            elif args.em == 'EM2':
                print(f'  向量尺度因子 k: [{avg_params[0]:.6f}, {avg_params[1]:.6f}, {avg_params[2]:.6f}]')
                f.write(f"向量尺度因子 k: [{avg_params[0]:.6f}, {avg_params[1]:.6f}, {avg_params[2]:.6f}]\n")
            elif args.em == 'EM3':
                print(f'  标量偏置 b: {avg_params[0]:.6f}')
                f.write(f"标量偏置 b: {avg_params[0]:.6f}\n")
            elif args.em == 'EM4':
                print(f'  向量偏置 b: [{avg_params[0]:.6f}, {avg_params[1]:.6f}, {avg_params[2]:.6f}]')
                f.write(f"向量偏置 b: [{avg_params[0]:.6f}, {avg_params[1]:.6f}, {avg_params[2]:.6f}]\n")
            elif args.em == 'EM5':
                k = avg_params[:3]
                b = avg_params[3:]
                print(f'  向量尺度因子 k: [{k[0]:.6f}, {k[1]:.6f}, {k[2]:.6f}]')
                print(f'  向量偏置 b: [{b[0]:.6f}, {b[1]:.6f}, {b[2]:.6f}]')
                f.write(f"向量尺度因子 k: [{k[0]:.6f}, {k[1]:.6f}, {k[2]:.6f}]\n")
                f.write(f"向量偏置 b: [{b[0]:.6f}, {b[1]:.6f}, {b[2]:.6f}]\n")
            elif args.em == 'EM':
                k = avg_params[:3]
                b = avg_params[3:]
                print(f'  比例因子 δk: [{k[0]:.6f}, {k[1]:.6f}, {k[2]:.6f}]')
                print(f'  三轴旋转误差 b: [{b[0]:.6f/pi*180}, {b[1]:.6f/pi*180}, {b[2]:.6f/pi*180}]（俯仰，偏航，横滚）')
                f.write(f"  比例因子 δk: [{k[0]:.6f}, {k[1]:.6f}, {k[2]:.6f}]\n")
                f.write(f"  三轴旋转误差 b: [{b[0]:.6f/pi*180}, {b[1]:.6f/pi*180}, {b[2]:.6f/pi*180}]（俯仰，偏航，横滚）\n")
            
            f.write("=" * 50 + "\n")
            f.write(f"训练完成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"训练总耗时: {time.time() - start_time:.2f} seconds\n")
            f.write(f"训练轮数: {args.epochs}\n")
            f.write(f"批次大小: {args.batch_size}\n")
            f.write(f"窗口大小: {args.window_size}\n")
            f.write(f"学习率: {args.lr}\n\n\n")
        
        print(f'\n标定参数已保存到 EM.txt 文件中')

    # 保存柱状图
    try:
        plt.figure(figsize=(3.6,3))
        plt.bar(['baseline','calibrated'], [base_rmse, cal_rmse], color=['#888', '#2b8a3e'])
        plt.ylabel('RMSE')
        plt.title(f'RMSE improvement {imp:.1f}%')
        out_bar = 'rmse_compare_'+f'{model.__class__.__name__}_Ultra_csv'+'.png'
        plt.tight_layout()
        plt.savefig(out_bar, dpi=160)
        print(f'Saved {out_bar}')
    except Exception as e:
        print(f'Plot bar failed: {e}')

    end_time = time.time()
    print(f'Total time: {end_time - start_time:.2f} seconds')

if __name__ == '__main__':
    main()


