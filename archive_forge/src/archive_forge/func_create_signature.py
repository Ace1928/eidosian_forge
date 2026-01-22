from collections import namedtuple
from textwrap import indent
from numba.types import float32, float64, int16, int32, int64, void, Tuple
from numba.core.typing.templates import signature
def create_signature(retty, args):
    """
    Given the return type and arguments for a libdevice function, return the
    signature of the stub function used to call it from CUDA Python.
    """
    return_types = [arg.ty for arg in args if arg.is_ptr]
    if retty != void:
        return_types.insert(0, retty)
    if len(return_types) > 1:
        retty = Tuple(return_types)
    else:
        retty = return_types[0]
    argtypes = [arg.ty for arg in args if not arg.is_ptr]
    return signature(retty, *argtypes)