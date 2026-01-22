import logging
import functools
import inspect
import itertools
import sys
import textwrap
import types
from pyomo.common.errors import DeveloperError
def relocated_module(new_name, msg=None, logger=None, version=None, remove_in=None):
    """Provide a deprecation path for moved / renamed modules

    Upon import, the old module (that called `relocated_module()`) will
    be replaced in `sys.modules` by an alias that points directly to the
    new module.  As a result, the old module should have only two lines
    of executable Python code (the import of `relocated_module` and the
    call to it).

    Parameters
    ----------
    new_name: str
        The new (fully-qualified) module name

    msg: str
        A custom deprecation message.

    logger: str
        The logger to use for emitting the warning (default: the calling
        pyomo package, or "pyomo")

    version: str [required]
        The version in which the module was renamed or moved.  General
        practice is to set version to the current development version
        (from `pyomo --version`) during development and update it to the
        actual release as part of the release process.

    remove_in: str
        The version in which the module will be removed from the code.

    Example
    -------
    >>> from pyomo.common.deprecation import relocated_module
    >>> relocated_module('pyomo.common.deprecation', version='1.2.3')
    WARNING: DEPRECATED: The '...' module has been moved to
        'pyomo.common.deprecation'. Please update your import.
        (deprecated in 1.2.3) ...

    """
    from importlib import import_module
    new_module = import_module(new_name)
    cf = _find_calling_frame(1)
    old_name = cf.f_globals.get('__name__', '<stdin>')
    cf = cf.f_back
    if cf is not None:
        importer = cf.f_back.f_globals['__name__'].split('.')[0]
        while cf is not None and cf.f_globals['__name__'].split('.')[0] == importer:
            cf = cf.f_back
    if cf is None:
        cf = _find_calling_frame(1)
    sys.modules[old_name] = new_module
    if msg is None:
        msg = f"The '{old_name}' module has been moved to '{new_name}'. Please update your import."
    deprecation_warning(msg, logger, version, remove_in, cf)