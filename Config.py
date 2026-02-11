import torch
# config.py



class Config:
    CUDA = True
    SEED = 42
    EPOCHS = 100
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    STRATIFY = True
    LEARNING_RATE = 0.001
    INPUT_DIM = 1044
    DATA_PATH = r"C:\Users\zhbshen\Desktop\数据集\5-SNV_hyperspectral.csv"
    TORCH_DTYPE = torch.float64
    SAVE_DIR = r"D:\Desktop\dac\DL325\20250409"
    BATCH_SIZE = 32
    NUM_CLASSES = 5  # 根据数据实际类别数初始化，后续会被覆盖
    hidden_dim = 64  # 模型隐藏层维度
    dropout_rate = 0.5  # 通用dropout率
    lstm_layers = 2  # LSTM层数
    transformer_heads = 4  # Transformer头数

config = Config()