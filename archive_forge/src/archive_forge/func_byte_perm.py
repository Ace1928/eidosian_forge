import functools
import os
from ..common.build import is_hip
from . import core
@core.extern
def byte_perm(arg0, arg1, arg2, _builder=None):
    return core.extern_elementwise('libdevice', libdevice_path(), [arg0, arg1, arg2], {(core.dtype('int32'), core.dtype('int32'), core.dtype('int32')): ('__nv_byte_perm', core.dtype('int32'))}, is_pure=True, _builder=_builder)