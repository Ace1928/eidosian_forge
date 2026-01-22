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
def wrap_outputs_maintaining_identity(outputs, unwrapped_inputs, orig_inputs, wrap_fn, out_dims=NO_OUT_DIMS):
    flat_unwrapped_inputs = pytree.arg_tree_leaves(*unwrapped_inputs)
    flat_orig_inputs = pytree.arg_tree_leaves(*orig_inputs)
    unwrapped_input_to_orig_input = {id(unwrapped): orig for unwrapped, orig in zip(flat_unwrapped_inputs, flat_orig_inputs)}
    flat_outputs, spec = pytree.tree_flatten(outputs)
    result = []
    out_dims_specified = out_dims != NO_OUT_DIMS
    if out_dims_specified:
        flat_out_dims = _broadcast_to_and_flatten(out_dims, spec)
        if flat_out_dims is None:
            raise RuntimeError(f"The autograd.Function's vmap staticmethod returned an incompatible (output, out_dims) tuple. Expected out_dims={out_dims} to be compatible with the structure of `output`. out_dims has structure {pytree.tree_flatten(out_dims)[1]} but output has structure {spec}. For more details, please see https://pytorch.org/docs/master/notes/extending.func.html")
    for i, output in enumerate(flat_outputs):
        if not isinstance(output, torch.Tensor):
            result.append(output)
            continue
        if id(output) in unwrapped_input_to_orig_input:
            result.append(unwrapped_input_to_orig_input[id(output)])
            continue
        if out_dims_specified:
            result.append(wrap_fn(output, flat_out_dims[i]))
        else:
            result.append(wrap_fn(output))
    return pytree.tree_unflatten(result, spec)