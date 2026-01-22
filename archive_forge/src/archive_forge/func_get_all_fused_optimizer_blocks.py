import collections
import itertools
import logging
import operator
import tempfile
import time
from dataclasses import dataclass, field
from functools import wraps
from typing import (
import torch
import torch.fx as fx
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.distributed._spmd.graph_utils import (
from torch.distributed._spmd.iter_graph_module import IterGraphModule
from torch.fx.passes.shape_prop import TensorMetadata
from torch.utils import _pytree as pytree
from torch.utils._pytree import tree_flatten, tree_unflatten
def get_all_fused_optimizer_blocks(gm: IterGraphModule, optim_ops: Union[Tuple[str, ...], str]) -> List[FusedOptimizerBlock]:
    """Find all the FusedOptimizerBlock that the optimizer operators are in `optim_ops`."""
    return [get_fused_optimizer_block(node) for node in gm.graph.nodes if node.name.startswith(optim_ops)]