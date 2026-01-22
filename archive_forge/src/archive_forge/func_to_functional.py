import contextlib
from abc import ABC, abstractmethod
from typing import Any, Callable, ContextManager, Tuple
import torch
import torch.utils._pytree as pytree
from torch._C import _functionalization_reapply_views_tls as _reapply_views
from torch.utils._python_dispatch import return_and_correct_aliasing, TorchDispatchMode
@staticmethod
def to_functional(x):
    assert not torch._is_functional_tensor(x)
    x_functional = torch._to_functional_tensor(x)
    with FunctionalTensorMode():
        torch._mirror_autograd_meta_to(x, x_functional)
        out = FunctionalTensor(x_functional)
        torch._mirror_autograd_meta_to(x_functional, out)
    return out