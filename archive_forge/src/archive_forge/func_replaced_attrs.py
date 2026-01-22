from __future__ import annotations
from typing import List, Optional, Tuple
import torch
import torch.fx
from torch.onnx._internal import _beartype
from torch.onnx._internal.fx import _pass
@property
def replaced_attrs(self) -> Tuple[torch.Tensor, ...]:
    """The list of replaced weight tensors."""
    assert self._replaced_attrs is not None, 'Must run ReplaceGetAttrWithPlaceholder first'
    return self._replaced_attrs