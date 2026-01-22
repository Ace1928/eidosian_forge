import functools
import warnings
from typing import Callable, Union
import torch
import torch.utils._pytree as pytree
from torch._ops import OpOverload
from torch._subclasses.fake_tensor import (
from torch.utils._python_dispatch import TorchDispatchMode
def is_sdpa_error(func, idx, e):
    if (func is aten._scaled_dot_product_flash_attention.default or func is aten._flash_attention_forward.default) and idx in (6, 7) and ('Devices' in repr(e)):
        return True
    if (func is aten._scaled_dot_product_efficient_attention.default or func is aten._efficient_attention_forward.default) and idx in (2, 3) and ('Devices' in repr(e)):
        return True
    return False