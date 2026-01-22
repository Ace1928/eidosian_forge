import os
import sys
from types import ModuleType
from .version import version as __version__  # NOQA:F401
def initpkg(pkgname, exportdefs, attr=None, eager=False):
    """ initialize given package from the export definitions. """
    attr = attr or {}
    oldmod = sys.modules.get(pkgname)
    d = {}
    f = getattr(oldmod, '__file__', None)
    if f:
        f = _py_abspath(f)
    d['__file__'] = f
    if hasattr(oldmod, '__version__'):
        d['__version__'] = oldmod.__version__
    if hasattr(oldmod, '__loader__'):
        d['__loader__'] = oldmod.__loader__
    if hasattr(oldmod, '__path__'):
        d['__path__'] = [_py_abspath(p) for p in oldmod.__path__]
    if hasattr(oldmod, '__package__'):
        d['__package__'] = oldmod.__package__
    if '__doc__' not in exportdefs and getattr(oldmod, '__doc__', None):
        d['__doc__'] = oldmod.__doc__
    d['__spec__'] = getattr(oldmod, '__spec__', None)
    d.update(attr)
    if hasattr(oldmod, '__dict__'):
        oldmod.__dict__.update(d)
    mod = ApiModule(pkgname, exportdefs, implprefix=pkgname, attr=d)
    sys.modules[pkgname] = mod
    if 'bpython' in sys.modules or eager:
        for module in list(sys.modules.values()):
            if isinstance(module, ApiModule):
                module.__dict__
    return mod