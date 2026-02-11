# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
import torch
from typing import Optional, Tuple, Dict
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, roc_curve,
    accuracy_score, f1_score, recall_score, precision_score,
    matthews_corrcoef
)
from sklearn.preprocessing import StandardScaler


# -----------------------------
# 随机种子
# -----------------------------
def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------
# 分类通用指标（支持二/多分类）
# y_prob: (N, C) 或 (N,)（二分类正类概率）
# -----------------------------
def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    average: str = "weighted",
) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    out = dict(
        ACC=float(accuracy_score(y_true, y_pred)),
        F1 =float(f1_score(y_true, y_pred, average=average, zero_division=0)),
        REC=float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        PRE=float(precision_score(y_true, y_pred, average=average, zero_division=0)),
    )

    if y_prob is not None:
        y_prob = np.asarray(y_prob)
        classes = np.unique(y_true)
        if classes.size == 2:
            # 二分类：允许 y_prob 为 (N, ) 或 (N, 2)
            if y_prob.ndim == 1:
                out["AUC"] = float(roc_auc_score(y_true, y_prob))
            else:
                out["AUC"] = float(roc_auc_score(y_true, y_prob[:, 1]))
        else:
            # 多分类：必须是 (N, C)
            if y_prob.ndim != 2 or y_prob.shape[1] != classes.size:
                # 不满足形状就不算 AUC，避免异常
                pass
            else:
                out["AUC"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr"))
    return out


# -----------------------------
# 二分类敏感度/特异度（传入0/1标签）
# -----------------------------
def sensitivity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(recall_score(y_true, y_pred, zero_division=0))

def specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    if cm.shape != (2,2):
        # 多分类场景下不定义
        return float("nan")
    tn, fp = cm[0, 0], cm[0, 1]
    denom = tn + fp
    return float(tn / denom) if denom else 1.0


# -----------------------------
# 二分类：基于 Youden 指数自动阈值，输出若干指标
# predict_prob: 正类概率 (N,)
# -----------------------------
def binary_threshold_metrics(y_true: np.ndarray, predict_prob: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    predict_prob = np.asarray(predict_prob).ravel()

    fpr, tpr, th = roc_curve(y_true, predict_prob)
    # Youden J = TPR - FPR
    best_idx = int(np.argmax(tpr - fpr))
    best_th = th[best_idx]

    y_pred = (predict_prob >= best_th).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    se = sensitivity(y_true, y_pred)
    sp = specificity(y_true, y_pred)

    return dict(
        threshold=float(best_th),
        AUC=float(roc_auc_score(y_true, predict_prob)),
        SEN=float(se),
        SPE=float(sp),
        PRE=float(precision_score(y_true, y_pred, zero_division=0)),
        ACC=float(accuracy_score(y_true, y_pred)),
        F1 =float(f1_score(y_true, y_pred, zero_division=0)),
        MCC=float(matthews_corrcoef(y_true, y_pred)),
        TN=int(cm[0,0]) if cm.size==4 else None,
        FP=int(cm[0,1]) if cm.size==4 else None,
        FN=int(cm[1,0]) if cm.size==4 else None,
        TP=int(cm[1,1]) if cm.size==4 else None,
    )


# -----------------------------
# 标准化：fit/transform 封装
# 注意：你原先的 scaler() 与 sklearn.StandardScaler 同名易混淆，
# 这里提供简单包装，避免命名冲突。
# -----------------------------
def fit_standard_scaler(X: np.ndarray) -> StandardScaler:
    sc = StandardScaler()
    sc.fit(X)
    return sc

def transform_with_scaler(X: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    return scaler.transform(X)

# 若你想用最小-最大归一化（0~1），提供这组函数：
def fit_minmax_scaler(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # 返回 (min, max)
    X = np.asarray(X, float)
    return X.min(axis=0), X.max(axis=0)

def transform_minmax(X: np.ndarray, vmin: np.ndarray, vmax: np.ndarray) -> np.ndarray:
    X = np.asarray(X, float)
    denom = np.where((vmax - vmin) == 0, 1.0, (vmax - vmin))
    return (X - vmin) / denom


# -----------------------------
# 图构造 & 归一化拉普拉斯
# -----------------------------
def knn_adjacency(
    feat: np.ndarray, k: int = 5, spectral_angle: bool = False
) -> np.ndarray:
    """
    基于特征列向量（feat: [N_samples, N_features]）构造 KNN 邻接矩阵（对称）。
    注意：这里以“样本”为点。若你希望“特征”为点，请先转置。
    """
    X = np.asarray(feat, float)
    # 计算余弦/欧氏距离的“相似度矩阵”简化实现
    # 先算 Gram 矩阵
    G = X @ X.T  # [N,N]
    nrm = np.sqrt(np.maximum(np.diag(G), 1e-12))
    if spectral_angle:
        # 1 - cos(theta) ≈ 1 - G/(||x||*||y||)
        S = 1.0 - (G / (np.outer(nrm, nrm) + 1e-12))
        D = S
        # 越小越近 -> 排序取最小的 k 个
        order = np.argsort(D, axis=1)
    else:
        # 欧氏距离的平方：||x||^2 + ||y||^2 - 2 x·y
        diag = np.diag(G)
        D = (diag[:, None] + diag[None, :] - 2.0 * G)
        order = np.argsort(D, axis=1)

    N = X.shape[0]
    A = np.zeros((N, N), dtype=float)
    for i in range(N):
        nbrs = order[i, 1 : k + 1]  # 跳过自己
        A[i, nbrs] = 1.0
    # 对称化
    A = np.maximum(A, A.T)
    return A


def normalized_laplacian(adj: np.ndarray) -> np.ndarray:
    """
    归一化拉普拉斯 L_sym = I - D^{-1/2} A D^{-1/2}
    """
    A = np.asarray(adj, float)
    deg = A.sum(axis=1)  # 度向量（按行求和）
    with np.errstate(divide="ignore"):
        d_inv_sqrt = 1.0 / np.sqrt(deg)
    d_inv_sqrt[~np.isfinite(d_inv_sqrt)] = 0.0

    D_inv_sqrt = np.diag(d_inv_sqrt)
    S = D_inv_sqrt @ A @ D_inv_sqrt
    N = A.shape[0]
    return np.eye(N) - S


# torch 版本的归一化拉普拉斯（避免创建大对角矩阵，走向量化）
def normalized_laplacian_torch(adj: torch.Tensor) -> torch.Tensor:
    """
    输入 adj: [N,N]（float）
    返回 L_sym: [N,N]
    """
    A = adj
    deg = A.sum(dim=1)                           # [N]
    d_inv_sqrt = torch.where(deg > 0, deg.rsqrt(), torch.zeros_like(deg))
    S = d_inv_sqrt[:, None] * A * d_inv_sqrt[None, :]
    N = A.shape[0]
    return torch.eye(N, dtype=A.dtype, device=A.device) - S


# -----------------------------
# 兼容你原先的 process_new（修复）
# 你之前写成了 scaler.transform(data) 但上面定义的是函数而不是对象。
# 这里提供一个安全的包装：传入你“拟合好的”StandardScaler或(min,max)元组。
# -----------------------------
def process_new(data: np.ndarray, scaler_obj) -> np.ndarray:
    """
    data: 新样本 (N, D)
    scaler_obj:
        - 若是 StandardScaler 实例：使用 .transform
        - 若是 (vmin, vmax) 元组：使用 0-1 归一化
    """
    if isinstance(scaler_obj, StandardScaler):
        return scaler_obj.transform(np.asarray(data, float))
    elif isinstance(scaler_obj, tuple) and len(scaler_obj) == 2:
        vmin, vmax = scaler_obj
        return transform_minmax(np.asarray(data, float), vmin, vmax)
    else:
        raise TypeError("scaler_obj 应为 sklearn.StandardScaler 或 (vmin, vmax) 元组")
