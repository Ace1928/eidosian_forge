from typing import Callable, Union, Tuple, List, Any, Optional
import torch
from functools import partial, wraps
import contextlib
from torch.utils._pytree import (
from torch.utils import _pytree as pytree
from torch.fx.experimental import const_fold
from torch.fx.experimental.proxy_tensor import make_fx
import torch.autograd.forward_ad as fwAD
from torch._subclasses.functional_tensor import FunctionalTensor
from .vmap import doesnt_support_saved_tensors_hooks, get_chunk_sizes
from .apis import vmap
from torch._C._functorch import (
from torch._functorch.utils import exposed_in, argnums_t
@exposed_in('torch.func')
def grad_and_value(func: Callable, argnums: argnums_t=0, has_aux: bool=False) -> Callable:
    """
    Returns a function to compute a tuple of the gradient and primal, or
    forward, computation.

    Args:
        func (Callable): A Python function that takes one or more arguments.
            Must return a single-element Tensor. If specified ``has_aux``
            equals ``True``, function can return a tuple of single-element
            Tensor and other auxiliary objects: ``(output, aux)``.
        argnums (int or Tuple[int]): Specifies arguments to compute gradients
            with respect to. ``argnums`` can be single integer or tuple of
            integers. Default: 0.
        has_aux (bool): Flag indicating that ``func`` returns a tensor and
            other auxiliary objects: ``(output, aux)``. Default: False.

    Returns:
        Function to compute a tuple of gradients with respect to its inputs
        and the forward computation. By default, the output of the function is
        a tuple of the gradient tensor(s) with respect to the first argument
        and the primal computation. If specified ``has_aux`` equals
        ``True``, tuple of gradients and tuple of the forward computation with
        output auxiliary objects is returned. If ``argnums`` is a tuple of
        integers, a tuple of a tuple of the output gradients with respect to
        each ``argnums`` value and the forward computation is returned.

    See :func:`grad` for examples
    """

    @doesnt_support_saved_tensors_hooks
    @wraps(func)
    def wrapper(*args, **kwargs):
        level = _grad_increment_nesting()
        try:
            output, aux, grad_input = (None, None, None)
            with torch.enable_grad():
                args = _wrap_all_tensors(args, level)
                kwargs = _wrap_all_tensors(kwargs, level)
                diff_args = _slice_argnums(args, argnums, as_tuple=False)
                tree_map_(partial(_create_differentiable, level=level), diff_args)
                output = func(*args, **kwargs)
                if has_aux:
                    if not (isinstance(output, tuple) and len(output) == 2):
                        raise RuntimeError('grad_and_value(f)(*args): output of function f should be a tuple: (output, aux) if has_aux is True')
                    output, aux = output
                if not isinstance(output, torch.Tensor):
                    raise RuntimeError(f'grad_and_value(f)(*args): Expected f(*args) to return a Tensor, got {type(output)}')
                if output.dim() != 0:
                    raise RuntimeError(f'grad_and_value(f)(*args): Expected f(*args) to return a scalar Tensor, got tensor with {output.dim()} dims. Maybe you wanted to use the vjp or jacrev APIs instead?')
                flat_diff_args, spec = tree_flatten(diff_args)
                flat_outputs = _as_tuple(output)
                flat_grad_input = _autograd_grad(flat_outputs, flat_diff_args, create_graph=True)
                grad_input = tree_unflatten(flat_grad_input, spec)
                grad_input = _undo_create_differentiable(grad_input, level)
                output = _undo_create_differentiable(output, level)
                if aux is not None:
                    aux = _undo_create_differentiable(aux, level)
            if has_aux:
                return (grad_input, (output, aux))
            return (grad_input, output)
        finally:
            _grad_decrement_nesting()
    return wrapper