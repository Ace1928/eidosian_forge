import functools
import os
from ..common.build import is_hip
from . import core
@core.extern
def min(arg0, arg1, _builder=None):
    arg0 = core._to_tensor(arg0, _builder)
    arg1 = core._to_tensor(arg1, _builder)
    arg0, arg1 = core.binary_op_type_legalization(arg0, arg1, _builder)
    dtype = arg0.dtype
    if dtype.is_floating():
        return core.tensor(_builder.create_minf(arg0.handle, arg1.handle), arg0.type)
    elif dtype.is_int_signed():
        return core.tensor(_builder.create_minsi(arg0.handle, arg1.handle), arg0.type)
    elif dtype.is_int_unsigned():
        return core.tensor(_builder.create_minui(arg0.handle, arg1.handle), arg0.dtype)
    else:
        assert False, f'Unexpected dtype {dtype}'