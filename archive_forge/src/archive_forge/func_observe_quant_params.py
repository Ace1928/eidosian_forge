import torch
from torch.nn.parameter import Parameter
from typing import List
@torch.jit.export
def observe_quant_params(self):
    print(f'_LearnableFakeQuantize Scale: {self.scale.detach()}')
    print(f'_LearnableFakeQuantize Zero Point: {self.zero_point.detach()}')