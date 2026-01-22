import sys
import marshal
import contextlib
import dis
from . import _imp
from ._imp import find_module, PY_COMPILED, PY_FROZEN, PY_SOURCE
from .extern.packaging.version import Version
def get_module_constant(module, symbol, default=-1, paths=None):
    """Find 'module' by searching 'paths', and extract 'symbol'

    Return 'None' if 'module' does not exist on 'paths', or it does not define
    'symbol'.  If the module defines 'symbol' as a constant, return the
    constant.  Otherwise, return 'default'."""
    try:
        f, path, (suffix, mode, kind) = info = find_module(module, paths)
    except ImportError:
        return None
    with maybe_close(f):
        if kind == PY_COMPILED:
            f.read(8)
            code = marshal.load(f)
        elif kind == PY_FROZEN:
            code = _imp.get_frozen_object(module, paths)
        elif kind == PY_SOURCE:
            code = compile(f.read(), path, 'exec')
        else:
            imported = _imp.get_module(module, paths, info)
            return getattr(imported, symbol, None)
    return extract_constant(code, symbol, default)