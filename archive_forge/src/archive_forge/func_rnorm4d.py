import functools
import os
from ..common.build import is_hip
from . import core
@core.extern
def rnorm4d(arg0, arg1, arg2, arg3, _builder=None):
    return core.extern_elementwise('libdevice', libdevice_path(), [arg0, arg1, arg2, arg3], {(core.dtype('fp32'), core.dtype('fp32'), core.dtype('fp32'), core.dtype('fp32')): ('__nv_rnorm4df', core.dtype('fp32')), (core.dtype('fp64'), core.dtype('fp64'), core.dtype('fp64'), core.dtype('fp64')): ('__nv_rnorm4d', core.dtype('fp64'))}, is_pure=True, _builder=_builder)