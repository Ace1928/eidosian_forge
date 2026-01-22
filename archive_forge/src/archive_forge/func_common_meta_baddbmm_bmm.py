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
def common_meta_baddbmm_bmm(batch1, batch2, is_bmm, self_baddbmm=None):
    torch._check(batch1.dim() == 3, lambda: 'batch1 must be a 3D tensor')
    torch._check(batch2.dim() == 3, lambda: 'batch2 must be a 3D tensor')
    batch1_sizes = batch1.size()
    batch2_sizes = batch2.size()
    bs = batch1_sizes[0]
    contraction_size = batch1_sizes[2]
    res_rows = batch1_sizes[1]
    res_cols = batch2_sizes[2]
    output_size = (bs, res_rows, res_cols)
    torch._check(batch2_sizes[0] == bs and batch2_sizes[1] == contraction_size, lambda: f'Expected size for first two dimensions of batch2 tensor to be: [{bs}, {contraction_size}] but got: [{batch2_sizes[0]}, {batch2_sizes[1]}].')
    output = batch2.new_empty(output_size)
    if not is_bmm and self_baddbmm is not None:
        torch._check(self_baddbmm.dim() == 3, lambda: 'self must be a 3D tensor')
        torch._check(self_baddbmm.size() == output_size, lambda: f'Expected an input tensor shape with shape {output_size} but got shape: {self_baddbmm.size()}')
    return output