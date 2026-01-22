import copy
import os
import sys
from importlib import import_module
from importlib.util import find_spec as importlib_find
def module_has_submodule(package, module_name):
    """See if 'module' is in 'package'."""
    try:
        package_name = package.__name__
        package_path = package.__path__
    except AttributeError:
        return False
    full_module_name = package_name + '.' + module_name
    try:
        return importlib_find(full_module_name, package_path) is not None
    except ModuleNotFoundError:
        return False