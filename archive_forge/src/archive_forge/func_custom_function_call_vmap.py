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
@custom_function_call.py_impl(TransformType.Vmap)
def custom_function_call_vmap(interpreter, autograd_function, *operands):
    if autograd_function.generate_vmap_rule:
        if has_overriden_vmap_rule(autograd_function):
            raise RuntimeError(f'You tried to vmap over {autograd_function.__name__}, but it has both generate_vmap_rule=True and an overriden vmap staticmethod. Please set generate_vmap_rule=False or delete the overriden vmap staticmethod to avoid ambiguity. For more details, please see https://pytorch.org/docs/master/notes/extending.func.html')
        return custom_function_call_vmap_generate_rule(interpreter, autograd_function, *operands)
    if not has_overriden_vmap_rule(autograd_function):
        raise RuntimeError(f'You tried to vmap over {autograd_function.__name__}, but it does not have vmap support. Please override and implement the vmap staticmethod or set generate_vmap_rule=True. For more details, please see https://pytorch.org/docs/master/notes/extending.func.html')
    current_level = interpreter.level()
    info = VmapInfo(batch_size=interpreter.batch_size(), randomness=interpreter.randomness())
    unwrapped_operands, in_dims = unwrap_batched(operands, current_level)
    if pytree.tree_all(lambda dim: dim is None, in_dims):
        with interpreter.lower():
            return custom_function_call(autograd_function, *operands)
    with interpreter.lower():
        result = autograd_function.vmap(info, in_dims, *unwrapped_operands)
    validate_vmap_returns_tuple_of_two_elements(result)
    unwrapped_output, out_dims = result

    def wrap_fn(output, out_dim):
        return output if out_dim is None else _add_batch_dim(output, out_dim, current_level)
    return wrap_outputs_maintaining_identity(unwrapped_output, unwrapped_operands, operands, wrap_fn, out_dims=out_dims)