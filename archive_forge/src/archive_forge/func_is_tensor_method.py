import inspect
import textwrap
import torch.jit
from torch.jit._builtins import _find_builtin
def is_tensor_method(schema):
    if len(schema.arguments) == 0:
        return False
    self = schema.arguments[0]
    if self.name != 'self':
        return False
    if not self.type.isSubtypeOf(torch._C.TensorType.get()):
        return False
    return True