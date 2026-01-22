import numbers
import warnings
import torch
import torch.nn as nn
from torch import Tensor  # noqa: F401
from torch._jit_internal import Tuple, Optional, List, Union, Dict  # noqa: F401
from torch.nn.utils.rnn import PackedSequence
from torch.ao.nn.quantized.modules.utils import _quantize_weight
def forward_impl(self, input: Tensor, hx: Optional[Tensor], batch_sizes: Optional[Tensor], max_batch_size: int, sorted_indices: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
    if hx is None:
        num_directions = 2 if self.bidirectional else 1
        zeros = torch.zeros(self.num_layers * num_directions, max_batch_size, self.hidden_size, dtype=input.dtype, device=input.device)
        hx = zeros
    else:
        hx = self.permute_hidden(hx, sorted_indices)
    self.check_forward_args(input, hx, batch_sizes)
    _all_params = [m.param for m in self._all_weight_values]
    if batch_sizes is None:
        result = torch.quantized_gru(input, hx, _all_params, self.bias, self.num_layers, self.dropout, self.training, self.bidirectional, self.batch_first)
    else:
        result = torch.quantized_gru(input, batch_sizes, hx, _all_params, self.bias, self.num_layers, self.dropout, self.training, self.bidirectional)
    output = result[0]
    hidden = result[1]
    return (output, hidden)