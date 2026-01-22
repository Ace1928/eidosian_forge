from typing import Any, Dict, List, Tuple
import torch
import torch.utils._pytree as pytree
from torch import Tensor
from torch._C import DispatchKey
from torch._ops import HigherOrderOperator
from torch._prims_common import clone_preserve_strides
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import (
def can_auto_functionalize(op: torch._ops.OperatorBase) -> bool:
    if not isinstance(op, torch._ops.OpOverload):
        return False
    if op.namespace == 'aten':
        return False
    schema = op._schema
    if not schema.is_mutable:
        return False
    if len(schema.returns) > 0:
        return False
    for arg in schema.arguments:
        if arg.alias_info is None:
            continue
        if not arg.alias_info.is_write:
            continue
        if type(arg.type) is torch.TensorType:
            continue
        if str(arg.type) == 'Optional[Tensor]':
            continue
        return False
    return True