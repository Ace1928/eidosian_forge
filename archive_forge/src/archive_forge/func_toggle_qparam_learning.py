import torch
from torch.nn.parameter import Parameter
from typing import List
@torch.jit.export
def toggle_qparam_learning(self, enabled=True):
    self.learning_enabled[0] = int(enabled)
    self.scale.requires_grad = enabled
    self.zero_point.requires_grad = enabled
    return self