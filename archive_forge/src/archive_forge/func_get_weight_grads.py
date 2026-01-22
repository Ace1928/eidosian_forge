import torch
import torch.nn as nn
from torch.distributed._shard.sharded_tensor import ShardedTensor
def get_weight_grads(self):
    return (self.fc1.weight.grad, self.fc2.weight.grad)