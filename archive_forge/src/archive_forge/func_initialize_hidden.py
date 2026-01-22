import numbers
from typing import Optional, Tuple
import warnings
import torch
from torch import Tensor
def initialize_hidden(self, batch_size: int, is_quantized: bool=False) -> Tuple[Tensor, Tensor]:
    h, c = (torch.zeros((batch_size, self.hidden_size)), torch.zeros((batch_size, self.hidden_size)))
    if is_quantized:
        h_scale, h_zp = self.initial_hidden_state_qparams
        c_scale, c_zp = self.initial_cell_state_qparams
        h = torch.quantize_per_tensor(h, scale=h_scale, zero_point=h_zp, dtype=self.hidden_state_dtype)
        c = torch.quantize_per_tensor(c, scale=c_scale, zero_point=c_zp, dtype=self.cell_state_dtype)
    return (h, c)