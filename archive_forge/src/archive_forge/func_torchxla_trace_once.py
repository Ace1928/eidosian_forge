import logging
import warnings
from functorch.compile import make_boxed_func
from ..backends.common import aot_autograd
from .registry import register_backend, register_experimental_backend
@register_experimental_backend
def torchxla_trace_once(model, fake_tensor_inputs):
    warnings.warn('This backend will be deprecated in 2.2, please use `openxla` backend instead')
    return xla_backend_helper(model, fake_tensor_inputs)