from __future__ import annotations
import abc
import collections
import copy
import operator
from typing import (
import torch
import torch.fx
from torch.onnx._internal import _beartype
from torch.onnx._internal.fx import _pass
from torch.utils import _pytree as pytree
@property
def raw_meta(self) -> Optional[Dict[str, Tuple[str, type]]]:
    """Returns the raw module stack meta data, i.e. node.meta['nn_module_stack']."""
    return {module_meta.raw_meta[0]: module_meta.raw_meta[1] for module_meta in self._module_stack}