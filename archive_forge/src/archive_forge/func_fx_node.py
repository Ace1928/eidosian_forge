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
def fx_node(self) -> torch.fx.Node:
    """Returns the fx.Node this instance represents."""
    return self._node