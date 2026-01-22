import math
from enum import Enum
from functools import partial
from typing import List, Optional, Sequence, Tuple, Union
import torch
import torch._prims_common as utils
from torch import SymBool, SymFloat, Tensor
from torch._decomp import (
from torch._ops import OpOverload
from torch._prims import _prim_elementwise_meta, ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND
from torch._prims_common import (
from torch._prims_common.wrappers import (
from torch._refs import _broadcast_shapes, _maybe_broadcast
from torch.utils import _pytree as pytree
import torch._refs
import torch._refs.nn.functional
import torch._refs.special
def checkLSTMBackwardSizes(grad_hy, grad_cy, cx, cy, workspace):
    defined_grad = grad_hy if grad_hy is not None else grad_cy
    torch._check(defined_grad.dim() == 2, lambda: '')
    exp_size = defined_grad.size()
    if grad_hy is not None:
        torch._check(grad_hy.size() == exp_size, lambda: '')
    if grad_cy is not None:
        torch._check(grad_cy.size() == exp_size, lambda: '')
    torch._check(cx.size() == exp_size, lambda: '')
    torch._check(cy.size() == exp_size, lambda: '')
    torch._check(workspace.dim() == 2, lambda: '')
    torch._check(workspace.numel() == exp_size[0] * exp_size[1] * 4, lambda: '')