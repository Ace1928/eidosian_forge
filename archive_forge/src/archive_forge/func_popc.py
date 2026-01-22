import functools
import os
from ..common.build import is_hip
from . import core
@core.extern
def popc(arg0, _builder=None):
    return core.extern_elementwise('libdevice', libdevice_path(), [arg0], {(core.dtype('int32'),): ('__nv_popc', core.dtype('int32')), (core.dtype('int64'),): ('__nv_popcll', core.dtype('int32'))}, is_pure=True, _builder=_builder)