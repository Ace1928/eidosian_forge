import contextlib
import functools
from llvmlite.ir import instructions, types, values
@_castop('fptrunc')
def fptrunc(self, value, typ, name=''):
    """
        Floating-point downcast to a less precise type:
            name = (typ) value
        """