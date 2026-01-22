import struct
from llvmlite.ir._utils import _StrCaching
def wrap_constant_value(self, values):
    from . import Value, Constant
    if not isinstance(values, (list, tuple)):
        return values
    if len(values) != len(self):
        raise ValueError('wrong constant size for %s: got %d elements' % (self, len(values)))
    return [Constant(ty, val) if not isinstance(val, Value) else val for ty, val in zip(self.elements, values)]