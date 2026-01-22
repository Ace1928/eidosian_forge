import torch
from torch._ops import HigherOrderOperator
from torch._C._functorch import TransformType
from torch._functorch.utils import enable_single_level_autograd_function
import torch.utils._pytree as pytree
from torch._C._functorch import (
from torch._functorch.vmap import (
from torch._functorch.apis import vmap
from torch._functorch.vmap import _broadcast_to_and_flatten
from torch.autograd.forward_ad import _set_fwd_grad_enabled
from typing import Any, NamedTuple, Tuple
def save_for_forward(self, *tensors):
    unwrapped_tensors, bdims = unwrap_batched(tensors, self._pt_current_level)
    self._pt_inner_ctx.save_for_forward(*unwrapped_tensors)
    self._pt_saved_tensors_bdims = bdims