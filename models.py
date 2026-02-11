#%% TorchWrapper
import time
from collections import defaultdict
from Config import config
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score


class BaseModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes


class TorchWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model_class, label_encoder):
        """
        PyTorch 模型包装器，兼容 scikit-learn API

        参数:
            model_class: 要实例化的模型类（例如 CNN）
            label_encoder: 训练数据使用的 LabelEncoder 实例
        """
        self.device = torch.device("cuda" if config.CUDA else "cpu")
        self.model = model_class(config.INPUT_DIM, config.NUM_CLASSES).to(self.device)
        self.label_encoder = label_encoder
        self.classes_ = label_encoder.classes_
        self.train_losses = []  # 全局训练损失
        self.class_losses = {  # 每个类别的损失记录
            cls: [] for cls in range(len(self.classes_))
        }
        self.num_classes = len(self.classes_)
        self.train_time = 0.0  # 添加训练时间记录
        self.epoch_metrics = defaultdict(list)  # 添加epoch指标记录

    def fit(self, X, y):
        """
        训练模型（兼容 scikit-learn 的 fit 方法）

        参数:
            X: 训练数据 (n_samples, n_features)
            y: 训练标签 (n_samples,)
        """
        start_time = time.time()  # 记录开始时间

        # 数据预处理
        X = self._ensure_float64(X)
        y = self._ensure_int64(y)

        # 转换为 PyTorch Dataset
        dataset = TensorDataset(
            torch.as_tensor(X, dtype=config.TORCH_DTYPE).unsqueeze(1),
            torch.as_tensor(y, dtype=torch.int64)
        )

        # 数据加载器
        loader = DataLoader(dataset,
                            batch_size=config.BATCH_SIZE,
                            shuffle=True,
                            pin_memory=config.CUDA)

        # 优化器设置
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.LEARNING_RATE
        )

        # 训练循环
        self.model.train()
        for epoch in range(config.EPOCHS):
            epoch_loss = 0.0
            class_epoch_loss = {cls: 0.0 for cls in range(self.num_classes)}
            class_count = {cls: 0 for cls in range(self.num_classes)}

            # 用于计算epoch指标
            all_outputs = []
            all_labels = []

            for batch_X, batch_y in loader:
                # 数据移动到设备
                batch_X = batch_X.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)

                # 前向传播
                optimizer.zero_grad()
                outputs = self.model(batch_X)

                # 计算损失
                loss = F.cross_entropy(outputs, batch_y)

                # 反向传播
                loss.backward()
                optimizer.step()

                # 记录损失
                epoch_loss += loss.item() * batch_X.size(0)

                # 计算每个类别的损失
                with torch.no_grad():
                    individual_loss = F.cross_entropy(
                        outputs,
                        batch_y,
                        reduction='none'
                    )

                    for cls in range(self.num_classes):
                        mask = (batch_y == cls)
                        if mask.sum() > 0:
                            class_epoch_loss[cls] += individual_loss[mask].sum().item()
                            class_count[cls] += mask.sum().item()

                # 收集指标数据
                all_outputs.append(outputs.detach().cpu())
                all_labels.append(batch_y.detach().cpu())

            # 记录 epoch 损失
            self.train_losses.append(epoch_loss / len(dataset))

            # 记录每个类别的平均损失
            for cls in range(self.num_classes):
                if class_count[cls] > 0:
                    self.class_losses[cls].append(
                        class_epoch_loss[cls] / class_count[cls]
                    )
                else:
                    self.class_losses[cls].append(0.0)

            # 计算epoch指标
            all_outputs = torch.cat(all_outputs)
            all_labels = torch.cat(all_labels)
            preds = torch.argmax(all_outputs, dim=1)

            # 计算并存储指标
            self.epoch_metrics['accuracy'].append(accuracy_score(all_labels, preds))
            self.epoch_metrics['f1'].append(f1_score(all_labels, preds, average='weighted'))
            self.epoch_metrics['recall'].append(recall_score(all_labels, preds, average='weighted'))
            self.epoch_metrics['precision'].append(precision_score(all_labels, preds, average='weighted'))

            # 打印训练进度
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{config.EPOCHS} | Loss: {self.train_losses[-1]:.4f} | "
                      f"Accuracy: {self.epoch_metrics['accuracy'][-1]:.4f}")

        # 记录总训练时间
        self.train_time = time.time() - start_time

        return self


    def predict_proba(self, X):
        """
        输出预测概率（兼容 scikit-learn API）

        参数:
            X: 输入数据 (n_samples, n_features)

        返回:
            numpy.ndarray: 预测概率 (n_samples, n_classes)
        """
        self.model.eval()
        X = self._ensure_float64(X)

        dataset = TensorDataset(
            torch.as_tensor(X, dtype=config.TORCH_DTYPE).unsqueeze(1)
        )
        loader = DataLoader(dataset,
                            batch_size=config.BATCH_SIZE,
                            shuffle=False,
                            pin_memory=config.CUDA)

        proba = []
        with torch.no_grad():
            for batch_X in loader:
                batch_X = batch_X[0].to(self.device, non_blocking=True)
                outputs = self.model(batch_X)
                proba.append(F.softmax(outputs, dim=1).cpu().numpy())

        return np.concatenate(proba, axis=0)

    def predict(self, X):
        """
        输出预测标签（兼容 scikit-learn API）

        参数:
            X: 输入数据 (n_samples, n_features)

        返回:
            numpy.ndarray: 预测标签 (n_samples,)
        """
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def save_model(self, path):
        """
        保存模型状态和元数据

        参数:
            path: 保存路径
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'class_names': self.label_encoder.classes_.tolist(),
            'input_dim': config.INPUT_DIM,
            'num_classes': config.NUM_CLASSES
        }, path)

    def _ensure_float64(self, data):
        """确保输入数据为 float64 类型"""
        if isinstance(data, np.ndarray) and data.dtype != np.float64:
            return data.astype(np.float64)
        return data

    def _ensure_int64(self, data):
        """确保标签数据为 int64 类型"""
        if isinstance(data, np.ndarray) and data.dtype != np.int64:
            return data.astype(np.int64)
        return data

    @property
    def _model_device(self):
        """返回模型所在设备信息（用于调试）"""
        return next(self.model.parameters()).device
    def predict_proba(self, X):
        """
        输出预测概率（兼容 scikit-learn API）

        参数:
            X: 输入数据 (n_samples, n_features)

        返回:
            numpy.ndarray: 预测概率 (n_samples, n_classes)
        """
        self.model.eval()
        X = self._ensure_float64(X)

        dataset = TensorDataset(
            torch.as_tensor(X, dtype=config.TORCH_DTYPE).unsqueeze(1)
        )
        loader = DataLoader(dataset,
                            batch_size=config.BATCH_SIZE,
                            shuffle=False,
                            pin_memory=config.CUDA)

        proba = []
        with torch.no_grad():
            for batch_X in loader:
                batch_X = batch_X[0].to(self.device, non_blocking=True)
                outputs = self.model(batch_X)
                proba.append(F.softmax(outputs, dim=1).cpu().numpy())

        return np.concatenate(proba, axis=0)

    def predict(self, X):
        """
        输出预测标签（兼容 scikit-learn API）

        参数:
            X: 输入数据 (n_samples, n_features)

        返回:
            numpy.ndarray: 预测标签 (n_samples,)
        """
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def save_model(self, path):
        """
        保存模型状态和元数据

        参数:
            path: 保存路径
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'class_names': self.label_encoder.classes_.tolist(),
            'input_dim': config.INPUT_DIM,
            'num_classes': config.NUM_CLASSES
        }, path)

    def _ensure_float64(self, data):
        """确保输入数据为 float64 类型"""
        if isinstance(data, np.ndarray) and data.dtype != np.float64:
            return data.astype(np.float64)
        return data

    def _ensure_int64(self, data):
        """确保标签数据为 int64 类型"""
        if isinstance(data, np.ndarray) and data.dtype != np.int64:
            return data.astype(np.int64)
        return data

    @property
    def _model_device(self):
        """返回模型所在设备信息（用于调试）"""
        return next(self.model.parameters()).device


#%% 模型--CNN
class CNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3, dtype=config.TORCH_DTYPE),
            nn.BatchNorm1d(64, dtype=config.TORCH_DTYPE),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(64, 128, kernel_size=5, padding=2, dtype=config.TORCH_DTYPE),
            nn.BatchNorm1d(128, dtype=config.TORCH_DTYPE),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(32)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 32, 256, dtype=config.TORCH_DTYPE),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(256, num_classes, dtype=config.TORCH_DTYPE)
        )
        self.double()

    def forward(self, x):
        assert x.dtype == config.TORCH_DTYPE, f"Input type error! Expected {config.TORCH_DTYPE}, got {x.dtype}"
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


#%% 模型--Transformer
class TransformerNet(BaseModel):
    def __init__(self, input_dim, num_classes):
        super().__init__(input_dim, num_classes)
        # 确保所有层初始化为double
        self.embed = nn.Linear(input_dim, config.hidden_dim)

        # Transformer配置
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=4,
            dim_feedforward=512,
            batch_first=True,  # 使用batch_first格式
            dtype=torch.float64  # 确保内部使用float64
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.classifier = nn.Linear(config.hidden_dim, num_classes)

        # 转换整个模型为double
        self.double()

    def forward(self, x):
        # 确保输入是float64且正确形状
        x = x.to(dtype=torch.float64)

        # 输入形状处理
        if x.dim() == 2:  # 如果是(batch_size, features)
            x = x.unsqueeze(1)  # 变成(batch_size, 1, features)
        elif x.dim() == 3 and x.size(1) == 1:  # 如果是(batch_size, 1, features)
            pass  # 已经是正确形状
        else:
            raise ValueError(f"输入形状错误，期望2D或3D(seq_len=1)，得到{x.shape}")

        # 通过模型层
        x = self.embed(x)  # (batch_size, 1, hidden_dim)
        x = self.transformer(x)  # (batch_size, 1, hidden_dim)
        x = x[:, -1, :]  # 取最后一个时间步 (batch_size, hidden_dim)
        return self.classifier(x)


#%%
class LSTM(BaseModel):  # 继承BaseModel（如果适用）
    def __init__(self, input_dim, num_classes):
        super().__init__(input_dim, num_classes)  # 如果BaseModel需要参数
        # LSTM配置（显式指定dtype=torch.float64）
        self.lstm1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=config.hidden_dim,
            batch_first=True,  # 使用batch_first格式
            dtype=torch.float64
        )
        self.lstm2 = nn.LSTM(
            input_size=config.hidden_dim,
            hidden_size=config.hidden_dim,
            batch_first=True,
            dtype=torch.float64
        )
        self.classifier = nn.Linear(
            config.hidden_dim,
            num_classes,
            dtype=torch.float64
        )

        # 确保整个模型为double
        self.double()

    def forward(self, x):
        # 确保输入是float64且正确形状
        x = x.to(dtype=torch.float64)

        # 输入形状处理（与TransformerNet一致）
        if x.dim() == 2:  # (batch_size, features)
            x = x.unsqueeze(1)  # (batch_size, 1, features)
        elif x.dim() == 3 and x.size(1) == 1:  # (batch_size, 1, features)
            pass  # 已经是正确形状
        else:
            raise ValueError(f"输入形状错误，期望2D或3D(seq_len=1)，得到{x.shape}")

        # 通过LSTM层
        x, _ = self.lstm1(x)  # (batch_size, 1, hidden_dim)
        x = torch.tanh(x)  # 激活函数

        x, _ = self.lstm2(x)  # (batch_size, 1, hidden_dim)
        x = torch.tanh(x)

        # 取最后一个时间步（与TransformerNet一致）
        x = x[:, -1, :]  # (batch_size, hidden_dim)

        # 分类层（根据需求选择是否加Sigmoid）
        return self.classifier(x)  # 或 torch.sigmoid(self.classifier(x))

#%%


class MLP(nn.Module):
    def __init__(self, input_dim=None, num_classes=None):
        super().__init__()
        # 动态获取配置（兼容您的config设置）
        self.input_dim = input_dim if input_dim else config.INPUT_DIM
        self.num_classes = num_classes if num_classes else config.NUM_CLASSES

        # 网络结构
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_classes)
        )

        # 训练记录
        self.class_losses = {i: [] for i in range(self.num_classes)}
        self.double()  # 确保双精度计算

    def forward(self, x):
        # 自动处理输入维度
        if x.dim() == 3:  # 处理可能的[batch, 1, features]输入
            x = x.squeeze(1)
        return self.net(x.double())

    def track_class_loss(self, loss, y):
        """记录各类别的损失（兼容您的主程序中的class_losses）"""
        with torch.no_grad():
            for cls in range(self.num_classes):
                mask = (y == cls)
                if mask.any():
                    self.class_losses[cls].append(loss[mask].mean().item())