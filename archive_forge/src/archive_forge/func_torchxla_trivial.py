import logging
import warnings
from functorch.compile import make_boxed_func
from ..backends.common import aot_autograd
from .registry import register_backend, register_experimental_backend
@register_experimental_backend
def torchxla_trivial(gm, fake_tensor_inputs):
    return gm