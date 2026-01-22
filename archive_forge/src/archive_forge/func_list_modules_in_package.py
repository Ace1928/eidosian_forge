import argparse
import pkgutil
import warnings
import types as pytypes
from numba.core import errors
from numba._version import get_versions
from numba.core.registry import cpu_target
from numba.tests.support import captured_stdout
def list_modules_in_package(package, module_filters=_default_module_filters):
    """Yield all modules in a given package.

    Recursively walks the package tree.
    """
    onerror_ignore = lambda _: None
    prefix = package.__name__ + '.'
    package_walker = pkgutil.walk_packages(package.__path__, prefix, onerror=onerror_ignore)

    def check_filter(modname):
        module_components = modname.split('.')
        return any((not filter_fn(module_components) for filter_fn in module_filters))
    modname = package.__name__
    if not check_filter(modname):
        yield package
    for pkginfo in package_walker:
        modname = pkginfo[1]
        if check_filter(modname):
            continue
        with captured_stdout():
            try:
                mod = __import__(modname)
            except Exception:
                continue
            for part in modname.split('.')[1:]:
                try:
                    mod = getattr(mod, part)
                except AttributeError:
                    mod = None
                    break
        if not isinstance(mod, pytypes.ModuleType):
            continue
        yield mod