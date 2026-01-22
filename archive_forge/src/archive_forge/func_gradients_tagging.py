import operator
from contextlib import contextmanager
from enum import Enum
from typing import Any, cast, Dict, List, Optional, Tuple
import torch
import torch.distributed.distributed_c10d as c10d
import torch.fx as fx
import torch.library
import torch.nn as nn
import torch.utils._pytree as pytree
from torch.distributed._spmd.batch_dim_utils import BatchDimAnalyzer
from torch.distributed._tensor import DeviceMesh, distribute_tensor, Replicate, Shard
from torch.distributed._tensor._utils import compute_local_shape
from torch.distributed._tensor.op_schema import (
from torch.distributed._tensor.placement_types import _Partial, DTensorSpec, Placement
from torch.distributed._tensor.redistribute import redistribute_local_tensor
from torch.fx import GraphModule
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.passes.shape_prop import _extract_tensor_metadata
from torch.nn.utils._named_member_accessor import NamedMemberAccessor
@contextmanager
def gradients_tagging(params: Dict[str, torch.Tensor]):
    """Tag the gradient of the parameters with a special tag, so that we can identify them during SPMD expansion.

    It's safe to trace those hooks and we would remove those nodes later.
    """
    tagging_hooks = []
    try:
        for p in params.values():
            h = p.register_hook(torch.ops._spmd.tag_grad)
            tagging_hooks.append(h)
        yield
    finally:
        for h in tagging_hooks:
            h.remove()