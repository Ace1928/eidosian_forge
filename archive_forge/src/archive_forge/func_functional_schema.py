import weakref
import torch
import torch.utils._pytree as pytree
from torch._C import _ExcludeDispatchKeyGuard, DispatchKey, DispatchKeySet
from torch._ops import OpOverload
from torch.library import Library
from torchgen.model import (
from .autograd import autograd_not_implemented
def functional_schema(new_op_name, op: OpOverload):
    schema = FunctionSchema.parse(str(op._schema))
    schema = schema.signature().with_name(OperatorName.parse(new_op_name))
    return str(schema)