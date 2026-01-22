import functools
import os
from ..common.build import is_hip
from . import core
@core.extern
def double2int_rz(arg0, _builder=None):
    return core.extern_elementwise('libdevice', libdevice_path(), [arg0], {(core.dtype('fp64'),): ('__nv_double2int_rz', core.dtype('int32'))}, is_pure=True, _builder=_builder)