from __future__ import annotations
import functools
import sys
import warnings
from typing import List, Optional, Sequence, Tuple, Union
import torch
import torch._C._onnx as _C_onnx
import torch.onnx
from torch import _C
from torch.onnx import (
from torch.onnx._globals import GLOBALS
from torch.onnx._internal import _beartype, jit_utils, registration
def to_slice_input(list_or_value, default_value=None):
    if is_none_value(list_or_value) and default_value is not None:
        list_or_value = [default_value]
    if isinstance(list_or_value, (list, torch.Tensor)):
        return g.op('Constant', value_t=torch.tensor(list_or_value))
    rank = symbolic_helper._get_tensor_rank(list_or_value)
    if rank == 0:
        return symbolic_helper._unsqueeze_helper(g, list_or_value, [0])
    if rank == 1:
        return list_or_value
    raise errors.SymbolicValueError(f'Rank must be 0 or 1, not {rank}', list_or_value)