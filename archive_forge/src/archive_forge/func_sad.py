import functools
import os
from ..common.build import is_hip
from . import core
@core.extern
def sad(arg0, arg1, arg2, _builder=None):
    return core.extern_elementwise('libdevice', libdevice_path(), [arg0, arg1, arg2], {(core.dtype('int32'), core.dtype('int32'), core.dtype('uint32')): ('__nv_sad', core.dtype('int32')), (core.dtype('uint32'), core.dtype('uint32'), core.dtype('uint32')): ('__nv_usad', core.dtype('uint32'))}, is_pure=True, _builder=_builder)