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
def generate_single_level_function(interpreter, autograd_function):
    level = interpreter.level()

    def forward(*operands):
        unwrapped_operands = pytree.tree_map_only(torch.Tensor, lambda x: _unwrap_for_grad(x, level), operands)
        with torch.enable_grad(), _set_fwd_grad_enabled(True), interpreter.lower():
            unwrapped_output = custom_function_call(autograd_function, *unwrapped_operands)

        def wrap_fn(output):
            return _wrap_for_grad(output, level)
        return wrap_outputs_maintaining_identity(unwrapped_output, unwrapped_operands, operands, wrap_fn)

    def setup_context(ctx, inputs, output):
        return autograd_function.setup_context(ctx, inputs, output)

    def backward(ctx, *grads):
        result = autograd_function.backward(ctx, *grads)
        return result

    def jvp(ctx, *tangents):
        result = autograd_function.jvp(ctx, *tangents)
        return result
    name = f'{autograd_function.__name__}Generated'
    Generated = type(name, (torch.autograd.function._SingleLevelFunction,), {'forward': staticmethod(forward), 'backward': staticmethod(backward), 'jvp': staticmethod(jvp), 'setup_context': staticmethod(setup_context)})
    return Generated