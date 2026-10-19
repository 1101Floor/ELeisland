import argparse
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch.nn.functional as F
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

np.random.seed(24)


# =========================================================
#  标准化 / 绘图 / Dataset / 时间特征 / 度量
# =========================================================
class StandardScaler:
    """train+valid 合并拟合的标准化器。

    fit() 由 train+valid 合并数据调用；test 与未来预测均使用同一份统计量，
    以保证 Train / Valid / Test 三份数据处于同一尺度，且 test 不参与拟合。
    每一列单独计算均值和总体标准差 (ddof=0)；预测目标固定在最后一列。
    """

    def __init__(self):
        self.mean_ = None
        self.std_ = None
        self.feature_columns_ = None

    def fit(self, fit_data, feature_columns):
        data = np.asarray(fit_data, dtype=np.float64)
        if data.ndim != 2 or data.shape[0] == 0:
            raise ValueError('拟合数据必须是非空的二维数组 [行数, 特征数]')
        if not np.isfinite(data).all():
            raise ValueError('拟合数据存在 NaN/Inf，无法计算可靠的标准化参数')
        if len(feature_columns) != data.shape[1]:
            raise ValueError('拟合数据列数与 feature_columns 不一致')

        self.mean_ = data.mean(axis=0)
        self.std_ = data.std(axis=0, ddof=0)
        self.std_[self.std_ < 1e-12] = 1.0
        self.feature_columns_ = tuple(feature_columns)
        return self

    @classmethod
    def load(cls, filepath, expected_columns):
        if not os.path.isfile(filepath):
            raise FileNotFoundError(
                f'未找到 train+valid 合并统计量文件: {filepath}。'
                '请先训练生成 train_scaler.npz，再执行独立测试/预测。'
            )
        with np.load(filepath, allow_pickle=False) as z:
            mean = z['mean'].astype(np.float64)
            std = z['std'].astype(np.float64)
            columns = z['feature_columns'].tolist()

        if columns != list(expected_columns):
            raise ValueError(
                '模型使用的特征列与训练时保存的标准化列不一致！\n'
                f'保存的列: {columns}\n当前的列: {list(expected_columns)}'
            )
        if mean.ndim != 1 or std.shape != mean.shape or len(mean) != len(columns):
            raise ValueError('train_scaler.npz 中标准化参数形状无效')
        if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
            raise ValueError('train_scaler.npz 包含无效的均值/标准差')

        obj = cls()
        obj.mean_ = mean
        obj.std_ = std
        obj.feature_columns_ = tuple(columns)
        return obj

    def save(self, filepath):
        if self.mean_ is None or self.std_ is None or self.feature_columns_ is None:
            raise RuntimeError('只能在 fit() 之后保存标准化参数')
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        np.savez_compressed(
            filepath, mean=self.mean_, std=self.std_,
            feature_columns=np.asarray(self.feature_columns_, dtype=str)
        )

    def _get_stats(self, data, target_only=False):
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError('标准化器未通过 fit() 或 load() 初始化')
        n_features = data.shape[-1]
        if target_only and n_features == 1:
            mean, std = self.mean_[-1:], self.std_[-1:]
        elif n_features == len(self.mean_):
            mean, std = self.mean_, self.std_
        else:
            raise ValueError(
                f'输入最后一维为 {n_features}，但训练统计量有 {len(self.mean_)} 个特征'
            )
        if torch.is_tensor(data):
            return (torch.as_tensor(mean, dtype=data.dtype, device=data.device),
                    torch.as_tensor(std, dtype=data.dtype, device=data.device))
        return mean, std

    def transform(self, data):
        mean, std = self._get_stats(data)
        out = (data - mean) / std
        if isinstance(data, np.ndarray):
            out = out.astype(data.dtype, copy=False)
        return out

    def inverse_transform(self, data):
        mean, std = self._get_stats(data, target_only=True)
        out = data * std + mean
        if isinstance(data, np.ndarray):
            out = out.astype(data.dtype, copy=False)
        return out


def plot_loss_data(data):
    plt.figure(figsize=(10, 5))
    plt.plot(data, marker='o')
    plt.title("Training Loss (standardized MSE, as optimized)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["Train Loss"])
    plt.grid(True)
    plt.show()

def plot_training_curves(history, save_path=None):
    if not history:
        return
    epochs = [h['epoch'] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ★ Loss 子图使用标准化空间（=训练时优化的 loss），不做反归一化
    axes[0].plot(epochs, [h['train_loss_standardized'] for h in history],
                 label='Training Loss')
    axes[0].plot(epochs, [h['valid_MSE_standardized'] for h in history],
                 label='Validation Loss ')
    axes[0].set_title('Loss Curve (Standardized Space, No Denormalization)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE Loss (standardized)')
    axes[0].legend(); axes[0].grid(True)

    axes[1].plot(epochs, [h['train_MAPE'] for h in history], label='Training MAPE')
    axes[1].plot(epochs, [h['valid_MAPE'] for h in history], label='Validation MAPE')
    axes[1].set_title('MAPE Curve')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('MAPE')
    axes[1].legend(); axes[1].grid(True)

    axes[2].plot(epochs, [h['train_DA'] for h in history], label='Training DA')
    axes[2].plot(epochs, [h['valid_DA'] for h in history], label='Validation DA')
    axes[2].set_title('DA Curve')
    axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('DA')
    axes[2].legend(); axes[2].grid(True)

    plt.tight_layout()
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, 'training_curves.png'),
                    dpi=300, bbox_inches='tight')
    plt.show()


class TimeSeriesDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        sequence, label, time_mark = self.sequences[index]
        return (torch.as_tensor(sequence, dtype=torch.float32),
                torch.as_tensor(label, dtype=torch.float32),
                torch.as_tensor(time_mark, dtype=torch.float32))


def build_time_features(dates, freq='d'):
    d = pd.DatetimeIndex(pd.to_datetime(dates, errors='raise'))
    feats = {
        'minute': d.minute.to_numpy() / 59.0 - 0.5,
        'hour': d.hour.to_numpy() / 23.0 - 0.5,
        'weekday': d.dayofweek.to_numpy() / 6.0 - 0.5,
        'monthday': (d.day.to_numpy() - 1) / 30.0 - 0.5,
        'yearday': (d.dayofyear.to_numpy() - 1) / 365.0 - 0.5,
        'second': d.second.to_numpy() / 59.0 - 0.5,
    }
    mapping = {
        'd': ['weekday', 'monthday', 'yearday'],
        'b': ['weekday', 'monthday', 'yearday'],
        'h': ['hour', 'weekday', 'monthday', 'yearday'],
        't': ['minute', 'hour', 'weekday', 'monthday', 'yearday'],
        'min': ['minute', 'hour', 'weekday', 'monthday', 'yearday'],
        's': ['second', 'minute', 'hour', 'weekday', 'monthday', 'yearday'],
    }
    if freq not in mapping:
        raise ValueError(f'暂不支持 freq={freq!r}；请选择 d/h/t/s')
    return np.stack([feats[k] for k in mapping[freq]], axis=-1).astype(np.float32)


def create_inout_sequences(input_data, time_mark, tw, pre_len, config):
    inout_seq = []
    for i in range(len(input_data) - tw - pre_len + 1):
        train_seq = input_data[i:i + tw]
        if config.feature in ['MS', 'S']:
            train_label = input_data[i + tw:i + tw + pre_len, -1:]
        else:
            train_label = input_data[i + tw:i + tw + pre_len]
        mark = time_mark[i:i + tw + pre_len]
        inout_seq.append((train_seq, train_label, mark))
    return inout_seq


def calculate_mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def evaluate_metrics(model, loader, scaler):
    """返回的业务指标（MAE/MSE/MAPE/DA）均在反归一化空间；
    另额外返回 MSE_standardized —— 标准化空间的 MSE，
    与训练时 loss_function 优化目标一致，用于 loss 曲线绘制与早停。

    ★ DA 统一为定义 B：预测值自身变化方向 vs 真值自身变化方向
      sign(pred[i]-pred[i-1]) == sign(true[i]-true[i-1])
    """
    was_training = model.training
    model.eval()

    predictions, truths = [], []
    preds_std_list, trues_std_list = [], []
    device = next(model.parameters()).device

    try:
        with torch.no_grad():
            for seq, label, time_mark in loader:
                pred = model(seq.to(device), x_mark_enc=time_mark.to(device))
                if isinstance(pred, tuple):
                    pred = pred[0]

                pred_raw = pred.detach().cpu().numpy()
                true_raw = label.detach().cpu().numpy()

                # 标准化空间（训练时优化的目标尺度）
                pred_std = pred_raw[..., -1]
                true_std = true_raw[..., -1]

                # 反归一化空间（业务指标）
                pred = scaler.inverse_transform(pred_raw)[..., -1]
                true = scaler.inverse_transform(true_raw)[..., -1]

                if pred.shape != true.shape:
                    raise ValueError('预测值和真实值形状不一致')

                predictions.append(pred)
                truths.append(true)
                preds_std_list.append(pred_std)
                trues_std_list.append(true_std)
    finally:
        model.train(was_training)

    if not predictions:
        raise ValueError('数据集没有滑动窗口，请检查数据量、seq_len 和 pre_len')

    pred = np.concatenate(predictions).astype(np.float64)
    true = np.concatenate(truths).astype(np.float64)
    pred_std = np.concatenate(preds_std_list).astype(np.float64)
    true_std = np.concatenate(trues_std_list).astype(np.float64)

    if not all(np.isfinite(a).all() for a in (pred, true, pred_std, true_std)):
        raise ValueError('数据或预测含 NaN/Inf，请检查缺失值')

    nonzero = np.abs(true) > 1e-8
    mape = float(
        np.mean(np.abs((pred[nonzero] - true[nonzero]) / true[nonzero]))
    ) if nonzero.any() else float('nan')

    # ---- DA：定义 B（预测值自身变化方向 vs 真值自身变化方向）----
    pred_flat = pred.reshape(-1)
    true_flat = true.reshape(-1)
    if pred_flat.size > 1:
        pred_trend = np.sign(pred_flat[1:] - pred_flat[:-1])
        true_trend = np.sign(true_flat[1:] - true_flat[:-1])
        da = float(np.mean(pred_trend == true_trend))
    else:
        da = float('nan')

    mse = float(np.mean((pred - true) ** 2))
    mse_standardized = float(np.mean((pred_std - true_std) ** 2))

    metrics = {
        'MAE': float(np.mean(np.abs(pred - true))),
        'MSE': mse,                               # 反归一化空间
        'MSE_standardized': mse_standardized,     # 标准化空间（与训练 loss 一致）
        'MAPE': mape,
        'DA': da,                                 # 定义 B
        'points': int(true.size),
        'MAPE_excluded_zero_targets': int((~nonzero).sum()),
    }
    return metrics, true, pred


# =========================================================
#   从三个独立 CSV 构建 dataloader
# =========================================================
def create_dataloader(config, device):
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>创建数据加载器<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")

    def _read_split(path, role):
        if path is None or not os.path.isfile(path):
            raise FileNotFoundError(f'[{role}] 数据文件不存在: {path}')
        df = pd.read_csv(path)
        if config.time_col is None:
            config.time_col = df.columns[0]
        if config.time_col not in df.columns:
            raise KeyError(f'[{role}] 没有找到时间列: {config.time_col}')
        if config.target == config.time_col:
            raise ValueError('时间列不能同时作为预测目标列')
        if config.target not in df.columns:
            raise KeyError(f'[{role}] 没有找到目标列: {config.target}')
        dates = pd.to_datetime(df[config.time_col], errors='raise')
        if dates.isna().any():
            raise ValueError(f'[{role}] 时间列有缺失值')
        if not dates.is_monotonic_increasing:
            raise ValueError(f'[{role}] 时间列必须按升序排列，请先按时间排序')
        return df, dates

    train_df, train_dates = _read_split(config.train_data_path, 'train')
    valid_df, valid_dates = _read_split(config.valid_data_path, 'valid')
    test_df,  test_dates  = _read_split(config.test_data_path,  'test')

    feature_cols = [c for c in train_df.columns if c not in (config.time_col, config.target)]
    if config.feature == 'S':
        feature_cols = []
    config.feature_columns = feature_cols + [config.target]
    config.input_size = len(config.feature_columns)

    for role, df in (('valid', valid_df), ('test', test_df)):
        missing = [c for c in config.feature_columns if c not in df.columns]
        if missing:
            raise KeyError(f'[{role}] 缺少列: {missing}')

    train_mark = build_time_features(train_dates, config.freq)
    valid_mark = build_time_features(valid_dates, config.freq)
    test_mark  = build_time_features(test_dates,  config.freq)

    train_raw = train_df[config.feature_columns].to_numpy(dtype=np.float32)
    valid_raw = valid_df[config.feature_columns].to_numpy(dtype=np.float32)
    test_raw  = test_df[config.feature_columns].to_numpy(dtype=np.float32)

    os.makedirs(config.save_path, exist_ok=True)
    scaler_path = os.path.join(config.save_path, 'train_scaler.npz')

    for role, raw in [('train', train_raw), ('valid', valid_raw), ('test', test_raw)]:
        if not np.isfinite(raw).all():
            raise ValueError(f'[{role}] 包含 NaN/Inf，请先处理缺失值')

    if getattr(config, 'train', True):
        # ★ train + valid 合并，统一计算归一化统计量（test 严格不参与 fit）
        fit_raw = np.concatenate([train_raw, valid_raw], axis=0)
        if not np.isfinite(fit_raw).all():
            raise ValueError('[train+valid] 包含 NaN/Inf，请先处理缺失值')
        scaler = StandardScaler().fit(fit_raw, config.feature_columns)
        scaler.save(scaler_path)
        pd.DataFrame({
            'Feature': config.feature_columns,
            'FitMean': scaler.mean_,
            'FitStd':  scaler.std_,
        }).to_csv(os.path.join(config.save_path, 'train_scaler_stats.csv'),
                  index=False, encoding='utf-8-sig')
        print(f'[Scaler] Train+Valid 合并拟合完成，统计量已保存: {scaler_path}')
    else:
        scaler = StandardScaler.load(scaler_path, config.feature_columns)
        print(f'[Scaler] 已加载训练阶段保存的合并统计量: {scaler_path}')

    # ★ 三份数据统一使用 train+valid 合并统计量做归一化
    train_norm = scaler.transform(train_raw)
    valid_norm = scaler.transform(valid_raw)
    test_norm  = scaler.transform(test_raw)
    print('[Scaler] Train / Valid / Test 统一使用 Train+Valid 合并的均值和标准差。')

    # ★★ 为 Val / Test 拼接前序数据作为上下文，从而预测 Val / Test 的全部点
    context_for_eval = getattr(config, 'context_for_eval', True)
    if context_for_eval:
        ctx_val = min(config.seq_len, len(train_raw))
        valid_norm_input = np.concatenate(
            [train_norm[-ctx_val:], valid_norm], axis=0)
        valid_mark_input = np.concatenate(
            [train_mark[-ctx_val:], valid_mark], axis=0)

        ctx_test = min(config.seq_len, len(valid_raw))
        test_norm_input = np.concatenate(
            [valid_norm[-ctx_test:], test_norm], axis=0)
        test_mark_input = np.concatenate(
            [valid_mark[-ctx_test:], test_mark], axis=0)

        print(f'[Context] Val 拼 Train 末 {ctx_val} 点作上下文；'
              f'Test 拼 Valid 末 {ctx_test} 点作上下文。'
              f'（可预测点数将覆盖 Val/Test 全量 {len(valid_raw)}/{len(test_raw)} 个点）')
    else:
        valid_norm_input = valid_norm
        valid_mark_input = valid_mark
        test_norm_input  = test_norm
        test_mark_input  = test_mark
        print('[Context] 关闭上下文拼接，Val/Test 将各自丢弃前 seq_len 个点。')

    train_seq = create_inout_sequences(train_norm, train_mark,
                                       config.seq_len, config.pre_len, config)
    valid_seq = create_inout_sequences(valid_norm_input, valid_mark_input,
                                       config.seq_len, config.pre_len, config)
    test_seq  = create_inout_sequences(test_norm_input,  test_mark_input,
                                       config.seq_len, config.pre_len, config)

    for role, seq in (('train', train_seq), ('valid', valid_seq), ('test', test_seq)):
        if not seq:
            raise ValueError(f'[{role}] 窗口不足，请减小 seq_len/pre_len 或增加 {role} 数据量')

    train_loader = DataLoader(TimeSeriesDataset(train_seq), batch_size=config.batch_size,
                              shuffle=config.shuffle, drop_last=False)
    valid_loader = DataLoader(TimeSeriesDataset(valid_seq), batch_size=config.batch_size,
                              shuffle=False, drop_last=False)
    test_loader  = DataLoader(TimeSeriesDataset(test_seq),  batch_size=config.batch_size,
                              shuffle=False, drop_last=False)

    print(f"训练集: 行数={len(train_df)}, 窗口数={len(train_seq)}")
    print(f"验证集: 行数={len(valid_df)}, 窗口数={len(valid_seq)}")
    print(f"测试集: 行数={len(test_df)},  窗口数={len(test_seq)}")
    print(f"特征数: {config.input_size}, 时间特征维数: {train_mark.shape[-1]}")

    return train_loader, test_loader, valid_loader, scaler


from BiSMambabock import BiSMambaBlock
from BiSMamba_Enc import Encoder, EncoderLayer
from forecast_embedding import DataEmbedding_wo_pos


class BiSMamba(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        self.enc_embedding = DataEmbedding_wo_pos(
            configs.enc_in, configs.d_model,
            configs.embed, configs.freq, configs.dropout
        )

        self.encoder = Encoder(
            [
                EncoderLayer(
                    BiSMambaBlock(d_model=configs.d_model, d_state=16, expand=1),
                    configs.d_model, configs.d_ff,
                    dropout=configs.dropout, activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        B, L, C = x_enc.shape
        if x_dec is None:
            x_dec = torch.zeros(B, self.pred_len, C,
                                device=x_enc.device, dtype=x_enc.dtype)
        x_enc = torch.cat([x_enc, x_dec[:, -self.pred_len:, :]], dim=1)
        if x_mark_enc is None:
            raise ValueError('时间嵌入已启用，必须传入包含历史及未来时间特征的 x_mark_enc')
        if x_mark_enc.shape[:2] != x_enc.shape[:2]:
            raise ValueError(f'时间特征长度错误: {x_mark_enc.shape} vs 输入 {x_enc.shape}')
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out)
        enc_out = self.projection(enc_out)
        if self.output_attention:
            return enc_out[:, -self.pred_len:, :], attns
        else:
            return enc_out[:, -self.pred_len:, :]


def load_best_checkpoint(args, device):
    """加载训练结束时的模型（早停触发或跑满 epochs 时保存的最后一轮）。"""
    p_best = os.path.join(args.save_path, 'best_model.pth')
    p_legacy = os.path.join(args.save_path, 'save_model.pth')

    if os.path.isfile(p_best):
        used_path = p_best
    elif os.path.isfile(p_legacy):
        used_path = p_legacy
        print(f"⚠ 未找到 {p_best}，回退到 {p_legacy}（无 epoch 元数据）")
    else:
        raise FileNotFoundError(
            f"未找到模型文件：{p_best} 或 {p_legacy}，请先执行训练。"
        )

    ckpt = torch.load(used_path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        return ckpt['state_dict'], ckpt, used_path
    else:
        print(f"⚠ 加载的是裸 state_dict（{used_path}），没有 epoch 元数据")
        return ckpt, None, used_path


# =========================================================
#   通用评估 + 输出预测图 & CSV（valid / test 复用）
# =========================================================
def evaluate_and_save_predictions(model, loader, scaler, device, save_path, split_name):
    """
    在给定 loader 上评估并保存预测图 (PNG) 与预测值/真实值 (CSV)。
    split_name: 'Valid' / 'Test' / 'Train'
    """
    os.makedirs(save_path, exist_ok=True)
    metrics, true, pred = evaluate_metrics(model, loader, scaler)

    labels_1d = np.asarray(true).reshape(len(true), -1)[:, 0]
    preds_1d = np.asarray(pred).reshape(len(pred), -1)[:, 0]

    # ---- 1) 预测图 ----
    plt.figure(figsize=(10, 5))
    plt.plot(labels_1d, label='TrueValue')
    plt.plot(preds_1d, label='Prediction')
    plt.title(f'{split_name} Set Prediction (Final Model)')
    plt.xlabel('Sample')
    plt.ylabel('Target Value')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(save_path, f'{split_name.lower()}_prediction.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.show()

    # ---- 2) 预测值 / 真实值 CSV（MAPE 为小数）----
    error = preds_1d - labels_1d
    abs_err = np.abs(error)
    with np.errstate(divide='ignore', invalid='ignore'):
        mape_pt = np.where(labels_1d != 0, np.abs(error / labels_1d), np.nan)

    df_pred = pd.DataFrame({
        'Index': np.arange(len(labels_1d)),
        'TrueValue': labels_1d,
        'Prediction': preds_1d,
        'Error': error,
        'AbsError': abs_err,
        'MAPE': mape_pt,
    })

    # ---- ★ 逐点方向：定义 B（预测自身趋势 vs 真值自身趋势）----
    if len(labels_1d) > 1:
        correct = (np.sign(preds_1d[1:] - preds_1d[:-1]) ==
                   np.sign(labels_1d[1:] - labels_1d[:-1])).astype(int)
        df_pred['CorrectDirection'] = np.concatenate([[np.nan], correct])
    else:
        df_pred['CorrectDirection'] = np.nan

    csv_path = os.path.join(save_path, f'{split_name.lower()}_predictions.csv')
    df_pred.to_csv(csv_path, index=False, encoding='utf-8-sig')

    # ★ 自检：CSV 逐点方向均值应与 metrics['DA'] 一致
    da_from_csv = float(np.nanmean(df_pred['CorrectDirection'].values)) \
        if len(df_pred) > 1 else float('nan')

    print(f"\n✔ {split_name} 集预测结果已保存:")
    print(f"  图像: {plot_path}")
    print(f"  CSV : {csv_path}")
    print(f"  样本数: {len(df_pred)}")
    print(f"  MAE:  {metrics['MAE']:.6f}")
    print(f"  MSE(raw): {metrics['MSE']:.6f}")
    print(f"  MSE(std): {metrics['MSE_standardized']:.6f}")
    print(f"  MAPE: {metrics['MAPE']:.6f}")
    print(f"  DA(metrics):     {metrics['DA']:.6f}")
    print(f"  DA(CSV 逐点均值): {da_from_csv:.6f}")

    return metrics, df_pred


# =========================================================
#   训练
# =========================================================
def train(model, args, scaler, device, train_loader, valid_loader):
    start_time = time.time()

    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    epochs = args.epochs
    patience = getattr(args, 'patience', 20)
    min_delta = getattr(args, 'min_delta', 0.0)

    results_loss = []
    epoch_metrics = []

    os.makedirs(args.save_path, exist_ok=True)

    best_valid_mse = float('inf')   # 现在追踪的是标准化空间 MSE
    epochs_no_improve = 0
    early_stopped = False
    actual_epochs = 0
    last_valid_metrics = None

    model_save_path = os.path.join(args.save_path, 'best_model.pth')
    legacy_save_path = os.path.join(args.save_path, 'save_model.pth')

    for i in tqdm(range(epochs)):
        actual_epochs = i + 1

        model.train()
        losss = []
        for seq, labels, time_mark in train_loader:
            optimizer.zero_grad()
            y_pred = model(seq.to(device), x_mark_enc=time_mark.to(device))
            single_loss = loss_function(y_pred, labels.to(device))
            single_loss.backward()
            optimizer.step()
            losss.append(single_loss.item())

        avg_loss = float(np.mean(losss))
        results_loss.append(avg_loss)

        train_metrics, _, _ = evaluate_metrics(model, train_loader, scaler)
        valid_metrics, _, _ = evaluate_metrics(model, valid_loader, scaler)
        last_valid_metrics = valid_metrics

        # ★ 早停使用标准化空间 MSE（与训练 loss 一致；由于反归一化是线性缩放，
        #   与反归一化 MSE 单调等价，行为不变，语义更清晰）
        current_val_mse_std = valid_metrics['MSE_standardized']
        current_mae_raw = valid_metrics['MAE']

        checkpoint = {
            'state_dict': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            'valid_metrics': valid_metrics.copy(),
            'epoch': i + 1,
        }
        torch.save(checkpoint, model_save_path)
        torch.save(
            {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            legacy_save_path
        )

        if np.isfinite(current_val_mse_std) and (current_val_mse_std < best_valid_mse - min_delta):
            best_valid_mse = current_val_mse_std
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        tqdm.write(
            f"Epoch {i + 1}/{epochs} | "
            f"Train Loss(std): {avg_loss:.6f} | "
            f"Valid Loss(std): {valid_metrics['MSE_standardized']:.6f} | "
            f"Train MSE(raw): {train_metrics['MSE']:.6f} | "
            f"Valid MSE(raw): {valid_metrics['MSE']:.6f} | "
            f"Train MAPE: {train_metrics['MAPE']:.4f} | "
            f"Valid MAPE: {valid_metrics['MAPE']:.4f} | "
            f"Train DA: {train_metrics['DA']:.4f} | "
            f"Valid DA: {valid_metrics['DA']:.4f} | "
            f"Valid MAE(raw): {current_mae_raw:.6f} | "
            f"Best Valid Loss(std): {best_valid_mse:.6f} "
            f"(patience: {epochs_no_improve}/{patience}) | "
            f"模型已保存"
        )

        epoch_metrics.append({
            'epoch': i + 1,
            'train_loss_standardized': avg_loss,
            'train_MSE_standardized': train_metrics['MSE_standardized'],
            'valid_MSE_standardized': valid_metrics['MSE_standardized'],
            'train_MSE': train_metrics['MSE'],       # raw 空间
            'valid_MSE': valid_metrics['MSE'],       # raw 空间
            'train_MAE': train_metrics['MAE'],
            'valid_MAE': valid_metrics['MAE'],
            'train_MAPE': train_metrics['MAPE'],
            'valid_MAPE': valid_metrics['MAPE'],
            'train_DA': train_metrics['DA'],
            'valid_DA': valid_metrics['DA'],
            'best_valid_MSE_standardized_so_far': best_valid_mse,
        })

        pd.DataFrame(epoch_metrics).to_csv(
            os.path.join(args.save_path, 'epoch_train_valid_metrics.csv'),
            index=False, encoding='utf-8-sig'
        )

        if epochs_no_improve >= patience:
            tqdm.write(
                f"\n*** Early stopping triggered at epoch {i + 1} ***\n"
                f"    验证集 Loss(std) 连续 {patience} 个 epoch 未改善（min_delta={min_delta}）\n"
                f"    已保存最后一轮（epoch {i + 1}）的模型"
            )
            early_stopped = True
            break

    if last_valid_metrics is None:
        raise RuntimeError("训练过程中未获得有效的验证集 MSE。")

    print("\n" + "=" * 60)
    print("训练结束，最后一轮模型指标（保存的即为此模型）")
    print("=" * 60)
    print(f"Last Epoch: {actual_epochs}")
    print(f"Last Valid MSE(std, 训练目标): {last_valid_metrics['MSE_standardized']:.6f}")
    print(f"Last Valid MSE(raw, 业务值):   {last_valid_metrics['MSE']:.6f}")
    print(f"Last Valid MAE(raw):           {last_valid_metrics['MAE']:.6f}")
    print(f"Last Valid MAPE:               {last_valid_metrics['MAPE']:.6f}")
    print(f"Last Valid DA (定义 B):        {last_valid_metrics['DA']:.6f}")
    print(f"Valid Points:                  {last_valid_metrics['points']}")
    print(f"（训练中历史最佳 Valid Loss(std): {best_valid_mse:.6f}，仅用于早停判断）")
    print("=" * 60)
    print(f"✔ 最终模型已保存至: {model_save_path}")

    if early_stopped:
        print(f"*** 早停触发: 实际训练 {actual_epochs} 个 epoch (patience={patience}) ***")
    else:
        print(f"训练完成全部 {epochs} 个 epoch，未触发早停。")

    last_result = {'last_epoch': actual_epochs, **last_valid_metrics}
    pd.DataFrame([last_result]).to_csv(
        os.path.join(args.save_path, 'last_valid_metrics.csv'),
        index=False, encoding='utf-8-sig'
    )

    print(f"模型训练完成，总用时: {(time.time() - start_time) / 60:.4f} min")

    plot_loss_data(results_loss)
    plot_training_curves(epoch_metrics, args.save_path)

    print("\n" + "=" * 60)
    print("加载最终模型（最后一轮）并输出 验证集(valid) 预测结果")
    print("=" * 60)
    state_dict, meta, used_path = load_best_checkpoint(args, device)
    model.load_state_dict(state_dict)
    evaluate_and_save_predictions(
        model, valid_loader, scaler, device, args.save_path, 'Valid'
    )

    return last_result


# =========================================================
#   test
# =========================================================
def test(model, args, test_loader, scaler):
    device = next(model.parameters()).device
    state_dict, meta, used_path = load_best_checkpoint(args, device)
    model.load_state_dict(state_dict)

    saved_metrics = meta.get('valid_metrics') if meta else None
    saved_epoch = meta.get('epoch') if meta else None

    print(f"已加载最终模型: {used_path}")
    print(f"  epoch={saved_epoch}")

    metrics, df_pred = evaluate_and_save_predictions(
        model, test_loader, scaler, device, args.save_path, 'Test'
    )

    print("\n" + "=" * 60)
    print("测试集指标（使用最后一轮训练保存的模型）")
    print("=" * 60)
    print(f"Last Epoch: {saved_epoch}")
    print(f"Test MAPE:     {metrics['MAPE']:.6f}")
    print(f"Test MAE(raw): {metrics['MAE']:.6f}")
    print(f"Test MSE(raw): {metrics['MSE']:.6f}")
    print(f"Test MSE(std): {metrics['MSE_standardized']:.6f}")
    print(f"Test DA (定义 B): {metrics['DA']:.6f}")
    print(f"Test Points:   {metrics['points']}")
    if saved_metrics is not None:
        print("-" * 60)
        print(f"(参考) 保存时 Valid MSE(std): {saved_metrics.get('MSE_standardized', float('nan')):.6f}")
        print(f"(参考) 保存时 Valid MSE(raw): {saved_metrics['MSE']:.6f}")
        print(f"(参考) 保存时 Valid MAE(raw): {saved_metrics['MAE']:.6f}")
        print(f"(参考) 保存时 Valid MAPE:     {saved_metrics['MAPE']:.6f}")
        print(f"(参考) 保存时 Valid DA:       {saved_metrics['DA']:.6f}")
    print("=" * 60)

    pd.DataFrame([metrics]).to_csv(
        os.path.join(args.save_path, 'test_metrics.csv'),
        index=False, encoding='utf-8-sig'
    )

    return metrics


def inspect_model_fit(model, args, train_loader, scaler):
    device = next(model.parameters()).device
    state_dict, meta, used_path = load_best_checkpoint(args, device)
    model.load_state_dict(state_dict)
    model.eval()

    epoch = meta.get('epoch') if meta else None
    print(f"[inspect_model_fit] 使用 {used_path}, epoch={epoch}")

    results, labels = [], []
    with torch.no_grad():
        for seq, label, time_mark in train_loader:
            pred = model(seq.to(device), x_mark_enc=time_mark.to(device))[:, 0, :]
            label = label[:, 0, :]
            pred = scaler.inverse_transform(pred.detach().cpu().numpy())
            label = scaler.inverse_transform(label.detach().cpu().numpy())
            for i in range(len(pred)):
                results.append(pred[i][-1])
                labels.append(label[i][-1])

    results = np.asarray(results, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)

    # ---- 保存 Train 拟合结果 CSV ----
    error = results - labels
    abs_err = np.abs(error)
    with np.errstate(divide='ignore', invalid='ignore'):
        mape_pt = np.where(labels != 0, np.abs(error / labels), np.nan)

    df_fit = pd.DataFrame({
        'Index': np.arange(len(labels)),
        'TrueValue': labels,
        'Prediction': results,
        'Error': error,
        'AbsError': abs_err,
        'MAPE': mape_pt,
    })

    # ---- ★ 逐点方向：定义 B（预测自身趋势 vs 真值自身趋势）----
    if len(labels) > 1:
        correct = (np.sign(results[1:] - results[:-1]) ==
                   np.sign(labels[1:] - labels[:-1])).astype(int)
        df_fit['CorrectDirection'] = np.concatenate([[np.nan], correct])
    else:
        df_fit['CorrectDirection'] = np.nan

    csv_path = os.path.join(args.save_path, 'train_fit_predictions.csv')
    df_fit.to_csv(csv_path, index=False, encoding='utf-8-sig')

    # ---- 汇总指标 ----
    mae = float(np.mean(abs_err))
    mse = float(np.mean(error ** 2))
    nonzero = np.abs(labels) > 1e-8
    mape = float(np.mean(np.abs(error[nonzero] / labels[nonzero]))) if nonzero.any() else float('nan')
    da = float(np.mean(df_fit['CorrectDirection'].dropna())) if len(labels) > 1 else float('nan')

    print(f"\n✔ Train 集拟合结果已保存:")
    print(f"  CSV : {csv_path}")
    print(f"  样本数: {len(df_fit)}")
    print(f"  MAE:  {mae:.6f}")
    print(f"  MSE:  {mse:.6f}")
    print(f"  MAPE: {mape:.6f}")
    print(f"  DA (定义 B): {da:.6f}")

    # ---- 拟合曲线图 ----
    plt.figure(figsize=(10, 5))
    plt.plot(labels, label='History')
    plt.plot(results, label='Prediction')
    plt.title(f"inspect model fit state (epoch={epoch})")
    plt.xlabel('Sample')
    plt.ylabel('Target Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.save_path, 'inspect_fit.png'),
                dpi=150, bbox_inches='tight')
    plt.show()

    return df_fit

def predict(model, args, device, scaler):
    df = pd.read_csv(args.predict_data_path)
    dates = pd.to_datetime(df[args.time_col], errors='raise')
    history_dates = pd.DatetimeIndex(dates.iloc[-args.seq_len:])
    if len(history_dates) < args.seq_len:
        raise ValueError('历史数据不足 seq_len')

    intervals = dates.diff().dropna()
    positive_intervals = intervals[intervals > pd.Timedelta(0)]
    if positive_intervals.empty:
        raise ValueError('无法确定未来时间步长，时间列需要至少两个递增时间点')
    step = positive_intervals.median()
    future_dates = pd.DatetimeIndex([history_dates[-1] + step * (i + 1)
                                     for i in range(args.pre_len)])
    all_dates = history_dates.append(future_dates)
    time_mark = torch.tensor(build_time_features(all_dates, args.freq),
                             dtype=torch.float32, device=device).unsqueeze(0)

    df_values = df[args.feature_columns].iloc[-args.seq_len:].to_numpy(dtype=np.float32)
    pre_data = scaler.transform(df_values)
    tensor_pred = torch.tensor(pre_data, dtype=torch.float32, device=device).unsqueeze(0)

    state_dict, meta, used_path = load_best_checkpoint(args, device)
    model.load_state_dict(state_dict)
    model.eval()

    epoch = meta.get('epoch') if meta else None
    print(f"[predict] 使用 {used_path}, epoch={epoch}")

    with torch.no_grad():
        pred = model(tensor_pred, x_mark_enc=time_mark)[0]
    pred = scaler.inverse_transform(pred.detach().cpu().numpy())

    future_csv = os.path.join(args.save_path, 'future_predictions.csv')
    pd.DataFrame({
        'Timestamp': future_dates.astype(str),
        'Prediction': pred[:, -1].tolist(),
    }).to_csv(future_csv, index=False, encoding='utf-8-sig')
    print(f"✔ 未来 {len(pred)} 步预测已保存至: {future_csv}")

    history_length = len(df_values)
    history_x = range(history_length)
    future_x = range(history_length, history_length + len(pred))
    plt.figure(figsize=(10, 5))
    plt.plot(history_x, df_values[:, -1], label='History')
    plt.plot([history_length - 1] + list(future_x),
             [df_values[-1, -1]] + pred[:, -1].tolist(), marker='o', label='Prediction')
    plt.axvline(history_length - 1, color='red')
    plt.xlabel('Sample Point')
    plt.title(f'History and Prediction (epoch={epoch})')
    plt.legend()
    plt.savefig(os.path.join(args.save_path, 'future_prediction.png'),
                dpi=150, bbox_inches='tight')


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in ('true', '1', 'yes', 'y'):
        return True
    if value in ('false', '0', 'no', 'n'):
        return False
    raise argparse.ArgumentTypeError('请输入 True 或 False')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Time series forecast')

    parser.add_argument('-model', type=str, default='Model')
    parser.add_argument('-seq_len', type=int, default=18)
    parser.add_argument('-pre_len', type=int, default=1)
    parser.add_argument('-time_col', type=str, default=None)
    parser.add_argument('-freq', type=str, default='d')
    parser.add_argument('-shuffle', action='store_true', default=False)

    parser.add_argument('-train_data_path', type=str, default='Train80.csv')
    parser.add_argument('-valid_data_path', type=str, default='Val20.csv')
    parser.add_argument('-test_data_path',  type=str, default='Test5.csv')
    parser.add_argument('-predict_data_path', type=str, default=None)

    parser.add_argument('-target', type=str, default='base_cost')
    parser.add_argument('-input_size', type=int, default=86)
    parser.add_argument('-feature', type=str, default='M')
    parser.add_argument('-lr', type=float, default=0.0001)
    parser.add_argument('-epochs', type=int, default=400)

    parser.add_argument('-patience', type=int, default=10)
    parser.add_argument('-min_delta', type=float, default=0.0)

    # ★ 是否给 Val/Test 拼接前序数据作为上下文（从而预测全量）
    parser.add_argument('-context_for_eval', type=str2bool, default=True,
                        help='True: Val 拼 Train 末尾 seq_len 点、Test 拼 Valid 末尾 seq_len 点，'
                             '从而预测 Val/Test 全部数据点；'
                             'False: 各自独立滑动窗口，丢弃前 seq_len 个点。')

    parser.add_argument('-lambda_diff', type=float, default=0)
    parser.add_argument('-batch_size', type=int, default=8)
    parser.add_argument('-save_path', type=str, default='models-11')
    parser.add_argument('-d_model', type=int, default=256)
    parser.add_argument('-d_ff', type=int, default=512)
    parser.add_argument('-dropout', type=float, default=0)
    parser.add_argument('-activation', type=str, default='relu')
    parser.add_argument('-e_layers', type=int, default=2)
    parser.add_argument('-use_gpu', type=str2bool, default=True)
    parser.add_argument('-device', type=int, default=0)
    parser.add_argument('-train', type=str2bool, default=False)
    parser.add_argument('-test', type=str2bool, default=True)
    parser.add_argument('-predict', type=str2bool, default=True)
    parser.add_argument('-inspect_fit', type=str2bool, default=True)
    parser.add_argument('-lr-scheduler', type=str2bool, default=True)

    args = parser.parse_args()

    if args.predict_data_path is None:
        args.predict_data_path = args.train_data_path

    if args.use_gpu and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device}")
    else:
        device = torch.device("cpu")
    print("使用设备:", device)

    train_loader, test_loader, valid_loader, scaler = create_dataloader(args, device)

    if args.feature in ['MS', 'S']:
        args.output_size = 1
    else:
        args.output_size = args.input_size

    args.pred_len = args.pre_len
    args.enc_in = args.input_size
    args.c_out = args.output_size
    args.output_attention = False
    args.embed = 'timeF'

    try:
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>开始初始化{args.model}模型<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        model = BiSMamba(args).to(device)
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>开始初始化{args.model}模型成功<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    except Exception as e:
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>开始初始化{args.model}模型失败<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        print("错误信息：", e)
        raise

    if args.train:
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>开始{args.model}模型训练<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        train(model, args, scaler, device, train_loader, valid_loader)

    if args.test:
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>开始{args.model}模型测试（测试集）<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        test(model, args, test_loader, scaler)

    if args.inspect_fit:
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>开始检验{args.model}模型拟合情况<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        inspect_model_fit(model, args, train_loader, scaler)

    if args.predict:
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>预测未来{args.pre_len}条数据<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        predict(model, args, device, scaler)

    plt.show()