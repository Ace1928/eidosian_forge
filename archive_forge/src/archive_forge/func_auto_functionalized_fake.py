from typing import Any, Dict, List, Tuple
import torch
import torch.utils._pytree as pytree
from torch import Tensor
from torch._C import DispatchKey
from torch._ops import HigherOrderOperator
from torch._prims_common import clone_preserve_strides
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import (
@auto_functionalized.py_impl(FakeTensorMode)
def auto_functionalized_fake(mode, op: torch._ops.OpOverload, mutated_args_names: List[str], kwargs: Dict[str, Any]) -> Tuple[Tensor, ...]:
    with mode:
        result = auto_functionalized_dense(op, mutated_args_names, kwargs)
        return result