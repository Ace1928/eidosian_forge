import contextlib
import functools
from llvmlite.ir import instructions, types, values
def convert_from_fp16(self, a, to=None, name=''):
    """
        Convert from an i16 to the given FP type
        """
    if not to:
        raise TypeError('expected a float return type')
    if not isinstance(to, (types.FloatType, types.DoubleType)):
        raise TypeError('expected a float type, got %s' % to)
    if not (isinstance(a.type, types.IntType) and a.type.width == 16):
        raise TypeError('expected an i16 type, got %s' % a.type)
    opname = 'llvm.convert.from.fp16'
    fn = self.module.declare_intrinsic(opname, [to])
    return self.call(fn, [a], name)