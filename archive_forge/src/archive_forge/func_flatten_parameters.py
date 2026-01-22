import math
import warnings
import numbers
import weakref
from typing import List, Tuple, Optional, overload
import torch
from torch import Tensor
from .module import Module
from ..parameter import Parameter
from ..utils.rnn import PackedSequence
from .. import init
from ... import _VF
def flatten_parameters(self) -> None:
    """Reset parameter data pointer so that they can use faster code paths.

        Right now, this works only if the module is on the GPU and cuDNN is enabled.
        Otherwise, it's a no-op.
        """
    if len(self._flat_weights) != len(self._flat_weights_names):
        return
    for w in self._flat_weights:
        if not isinstance(w, Tensor):
            return
    first_fw = self._flat_weights[0]
    dtype = first_fw.dtype
    for fw in self._flat_weights:
        if not isinstance(fw.data, Tensor) or not fw.data.dtype == dtype or (not fw.data.is_cuda) or (not torch.backends.cudnn.is_acceptable(fw.data)):
            return
    unique_data_ptrs = {p.data_ptr() for p in self._flat_weights}
    if len(unique_data_ptrs) != len(self._flat_weights):
        return
    with torch.cuda.device_of(first_fw):
        import torch.backends.cudnn.rnn as rnn
        with torch.no_grad():
            if torch._use_cudnn_rnn_flatten_weight():
                num_weights = 4 if self.bias else 2
                if self.proj_size > 0:
                    num_weights += 1
                torch._cudnn_rnn_flatten_weight(self._flat_weights, num_weights, self.input_size, rnn.get_cudnn_mode(self.mode), self.hidden_size, self.proj_size, self.num_layers, self.batch_first, bool(self.bidirectional))