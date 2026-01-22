import warnings
from typing import List, Optional, Tuple
import torch
from torch import _VF, Tensor  # noqa: F401
from torch.nn.utils.rnn import PackedSequence
def quantize_rnn_cell_modules(module):
    warnings.warn('quantize_rnn_cell_modules function has been deprecated. Please use torch.ao.quantization.quantize_dynamic API instead.')
    reassign = {}
    for name, mod in module.named_modules():
        if mod is module:
            continue
        new_mod = quantize_rnn_cell_modules(mod)
        if new_mod is not mod:
            reassign[name] = new_mod
    for name, mod in reassign.items():
        setattr(module, name, mod)
    if isinstance(module, torch.nn.LSTMCell):
        return QuantizedLSTMCell(module)
    if isinstance(module, torch.nn.GRUCell):
        return QuantizedGRUCell(module)
    if isinstance(module, torch.nn.RNNCell):
        return QuantizedRNNCell(module)
    return module