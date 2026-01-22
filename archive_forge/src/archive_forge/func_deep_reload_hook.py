imported from that module, which is useful when you're changing files deep
import builtins as builtin_mod
from contextlib import contextmanager
import importlib
import sys
from types import ModuleType
from warnings import warn
import types
def deep_reload_hook(m):
    """Replacement for reload()."""
    if m is types:
        return m
    if not isinstance(m, ModuleType):
        raise TypeError('reload() argument must be module')
    name = m.__name__
    if name not in sys.modules:
        raise ImportError('reload(): module %.200s not in sys.modules' % name)
    global modules_reloading
    try:
        return modules_reloading[name]
    except:
        modules_reloading[name] = m
    try:
        newm = importlib.reload(m)
    except:
        sys.modules[name] = m
        raise
    finally:
        modules_reloading.clear()
    return newm