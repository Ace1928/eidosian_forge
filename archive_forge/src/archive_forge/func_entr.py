import math
from typing import Optional, Union
import torch
import torch._prims as prims
import torch._prims_common as utils
import torch._refs as refs
from torch import Tensor
from torch._decomp import register_decomposition
from torch._prims_common import (
from torch._prims_common.wrappers import elementwise_type_promotion_wrapper, out_wrapper
from torch._refs import (
@register_decomposition(aten.special_entr)
@out_wrapper()
@elementwise_type_promotion_wrapper(type_promoting_args=('a',), type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def entr(a: TensorLikeType) -> TensorLikeType:
    return torch.where(torch.isnan(a), a, torch.where(a > 0, -a * torch.log(a), torch.where(a == 0, 0, -torch.inf)))