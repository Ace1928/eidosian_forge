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
def module_display_name(self) -> str:
    """Returns the module display name of the top module."""
    return self.top().module_display_name