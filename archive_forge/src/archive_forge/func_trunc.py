import functools
import os
from ..common.build import is_hip
from . import core
@core.extern
def trunc(arg0, _builder=None):
    return core.extern_elementwise('libdevice', libdevice_path(), [arg0], {(core.dtype('fp64'),): ('__nv_trunc', core.dtype('fp64')), (core.dtype('fp32'),): ('__nv_truncf', core.dtype('fp32'))}, is_pure=True, _builder=_builder)