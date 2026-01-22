import logging
import functools
import inspect
import itertools
import sys
import textwrap
import types
from pyomo.common.errors import DeveloperError
def relocated_module_attribute(local, target, version, remove_in=None, msg=None, f_globals=None):
    """Provide a deprecation path for moved / renamed module attributes

    This function declares that a local module attribute has been moved
    to another location.  For Python 3.7+, it leverages a
    module.__getattr__ method to manage the deferred import of the
    object from the new location (on request), as well as emitting the
    deprecation warning.

    Parameters
    ----------
    local: str
        The original (local) name of the relocated attribute

    target: str
        The new absolute import name of the relocated attribute

    version: str
        The Pyomo version when this move was released
        (passed to deprecation_warning)

    remove_in: str
        The Pyomo version when this deprecation path will be removed
        (passed to deprecation_warning)

    msg: str
        If not None, then this specifies a custom deprecation message to
        be emitted when the attribute is accessed from its original
        location.

    """
    if version is None:
        raise DeveloperError("relocated_module_attribute(): missing 'version' argument")
    if f_globals is None:
        f_globals = inspect.currentframe().f_back.f_globals
        if f_globals['__name__'].startswith('importlib.'):
            raise DeveloperError('relocated_module_attribute() called from a cythonized module without passing f_globals')
    _relocated = f_globals.get('__relocated_attrs__', None)
    if _relocated is None:
        f_globals['__relocated_attrs__'] = _relocated = {}
        _mod_getattr = f_globals.get('__getattr__', None)

        def __getattr__(name):
            info = _relocated.get(name, None)
            if info is not None:
                target_obj = _import_object(name, *info)
                f_globals[name] = target_obj
                return target_obj
            elif _mod_getattr is not None:
                return _mod_getattr(name)
            raise AttributeError("module '%s' has no attribute '%s'" % (f_globals['__name__'], name))
        f_globals['__getattr__'] = __getattr__
    _relocated[local] = (target, version, remove_in, msg)