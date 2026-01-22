import numbers
import warnings
import torch
import torch.nn as nn
from torch import Tensor  # noqa: F401
from torch._jit_internal import Tuple, Optional, List, Union, Dict  # noqa: F401
from torch.nn.utils.rnn import PackedSequence
from torch.ao.nn.quantized.modules.utils import _quantize_weight
@torch.jit.export
def forward_tensor(self, input: Tensor, hx: Optional[Tensor]=None) -> Tuple[Tensor, Tensor]:
    batch_sizes = None
    max_batch_size = input.size(0) if self.batch_first else input.size(1)
    sorted_indices = None
    unsorted_indices = None
    output, hidden = self.forward_impl(input, hx, batch_sizes, max_batch_size, sorted_indices)
    return (output, self.permute_hidden(hidden, unsorted_indices))