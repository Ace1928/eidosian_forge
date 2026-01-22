import functools
import os
from ..common.build import is_hip
from . import core
@core.extern
def div_rz(arg0, arg1, _builder=None):
    return core.extern_elementwise('libdevice', libdevice_path(), [arg0, arg1], {(core.dtype('fp32'), core.dtype('fp32')): ('__nv_fdiv_rz', core.dtype('fp32')), (core.dtype('fp64'), core.dtype('fp64')): ('__nv_ddiv_rz', core.dtype('fp64'))}, is_pure=True, _builder=_builder)