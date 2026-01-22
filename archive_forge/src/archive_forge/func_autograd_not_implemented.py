import torch
import torch.utils._pytree as pytree
from collections import namedtuple
import functools
def autograd_not_implemented(custom_op):

    def kernel(*args, **kwargs):
        if torch.is_grad_enabled() and pytree.tree_any(lambda x: isinstance(x, torch.Tensor) and x.requires_grad, (args, kwargs)):
            raise RuntimeError('Autograd has not been implemented for operator')
        with torch._C._AutoDispatchBelowAutograd():
            return custom_op(*args, **kwargs)
    return kernel