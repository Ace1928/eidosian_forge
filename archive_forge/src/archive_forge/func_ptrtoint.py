import contextlib
import functools
from llvmlite.ir import instructions, types, values
@_castop('ptrtoint')
def ptrtoint(self, value, typ, name=''):
    """
        Cast pointer to integer:
            name = (typ) value
        """