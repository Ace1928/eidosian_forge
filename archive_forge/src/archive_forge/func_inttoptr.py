import contextlib
import functools
from llvmlite.ir import instructions, types, values
@_castop('inttoptr')
def inttoptr(self, value, typ, name=''):
    """
        Cast integer to pointer:
            name = (typ) value
        """