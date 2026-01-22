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
def is_parent_module_of(self, node: _IRNode) -> bool:
    """Determines if this node represents a parent module of the provided node."""
    return node.stack_meta.is_superset_of(self.stack_meta)