import gc
import torch
from torch.utils import _pytree
from ._utils import _dummy_type
from torch._C import (  # noqa: F401
def functionalized(*user_args):
    flatten_user_args = _pytree.arg_tree_leaves(*user_args)
    out = Graphed.apply(*tuple(flatten_user_args) + module_params)
    return _pytree.tree_unflatten(out, output_unflatten_spec)