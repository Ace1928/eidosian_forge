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
def fx_nodes(self) -> Generator[torch.fx.Node, None, None]:
    """Returns an iterator for the sequence of fx nodes this instance holds."""
    for node in self._nodes:
        if isinstance(node, _ModuleNode):
            yield from node.fx_nodes()
        else:
            assert isinstance(node, _LeafNode)
            yield node.fx_node