import torch.nn as nn
from fairscale.optim import GradScaler
def get_benchmark_config():
    return {'epochs': 1, 'lr': 0.001, 'batch_size': 32, 'criterion': nn.CrossEntropyLoss()}