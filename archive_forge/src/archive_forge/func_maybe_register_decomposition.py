import inspect
from typing import Callable, Dict, List, Optional, Tuple
import torch
import torch._decomp
from torch import Tensor
from torch._prims_common.wrappers import _maybe_remove_out_wrapper
def maybe_register_decomposition(op):

    def decorator(f):
        try:
            return register_decomposition(op)(f)
        except Exception:
            return f
    return decorator