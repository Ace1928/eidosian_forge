import gc
import torch
from torch.utils import _pytree
from ._utils import _dummy_type
from torch._C import (  # noqa: F401
def make_graphed_forward(func, graph_training_state, graphed, orig_fwd):

    def new_fwd(*user_args):
        if func.training == graph_training_state:
            return graphed(*user_args)
        else:
            return orig_fwd(*user_args)
    return new_fwd