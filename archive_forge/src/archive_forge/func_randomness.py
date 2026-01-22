from abc import ABC, abstractmethod
import contextlib
from typing import Any
import torch
import torch.utils._pytree as pytree
from torch._C._functorch import (
from torch.autograd.forward_ad import _set_fwd_grad_enabled
def randomness(self):
    typ = self._cptr.randomness()
    if typ == RandomnessType.Error:
        return 'error'
    elif typ == RandomnessType.Same:
        return 'same'
    elif typ == RandomnessType.Different:
        return 'different'
    raise RuntimeError(f'Unknown RandomnessType: {typ}')