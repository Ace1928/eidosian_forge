import collections
import functools
import warnings
from itertools import product
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union
import torch
import torch.testing
from torch._vmap_internals import _vmap, vmap
from torch.overrides import is_tensor_like
from torch.types import _TensorOrTensors
def get_analytical_jacobian(inputs, output, nondet_tol=0.0, grad_out=1.0):
    warnings.warn("get_analytical_jacobian was part of PyTorch's private API and not meant to be exposed. We are deprecating it and it will be removed in a future version of PyTorch. If you have a specific use for this or feature request for this to be a stable API, please file us an issue at https://github.com/pytorch/pytorch/issues/new")
    if grad_out != 1.0:
        raise ValueError('Expected grad_out to be 1.0. get_analytical_jacobian no longer supports values of grad_out != 1.0.')
    if output.is_complex():
        raise ValueError('Expected output to be non-complex. get_analytical_jacobian no longer supports functions that return complex outputs.')
    diff_input_list = list(_iter_tensors(inputs, True))

    def vjp_fn(grad_output):
        return torch.autograd.grad(output, diff_input_list, grad_output, retain_graph=True, allow_unused=True)
    vjps1 = _compute_analytical_jacobian_rows(vjp_fn, output.clone())
    vjps2 = _compute_analytical_jacobian_rows(vjp_fn, output.clone())
    output_numel = output.numel()
    jacobians1, types_ok, sizes_ok = _stack_and_check_tensors(vjps1, inputs, output_numel)
    jacobians2, _, _ = _stack_and_check_tensors(vjps2, inputs, output_numel)
    reentrant = _check_jacobians_equal(jacobians1, jacobians2, nondet_tol)
    return (jacobians1, reentrant, sizes_ok, types_ok)