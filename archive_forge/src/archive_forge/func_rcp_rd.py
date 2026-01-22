import functools
import os
from ..common.build import is_hip
from . import core
@core.extern
def rcp_rd(arg0, _builder=None):
    return core.extern_elementwise('libdevice', libdevice_path(), [arg0], {(core.dtype('fp32'),): ('__nv_frcp_rd', core.dtype('fp32')), (core.dtype('fp64'),): ('__nv_drcp_rd', core.dtype('fp64'))}, is_pure=True, _builder=_builder)