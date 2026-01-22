import functools
import os
from ..common.build import is_hip
from . import core
@core.extern
def float2ll_rz(arg0, _builder=None):
    return core.extern_elementwise('libdevice', libdevice_path(), [arg0], {(core.dtype('fp32'),): ('__nv_float2ll_rz', core.dtype('int64'))}, is_pure=True, _builder=_builder)