import contextlib
import functools
import logging
from unittest.mock import patch
import torch
from torch._dynamo import disable
from torch._dynamo.utils import counters, defake
from torch._functorch.aot_autograd import aot_module_simplified
from torch.utils._python_dispatch import _disable_current_modes
def fake_tensor_unsupported(fn):
    """
    Decorator for backends that need real inputs.  We swap out fake
    tensors for zero tensors.
    """

    @functools.wraps(fn)
    def wrapper(model, inputs, **kwargs):
        with _disable_current_modes():
            inputs = list(map(defake, inputs))
            return fn(model, inputs, **kwargs)
    return wrapper