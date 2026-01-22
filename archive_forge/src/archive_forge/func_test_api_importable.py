import pkgutil
import types
import importlib
import warnings
from importlib import import_module
import pytest
import scipy
def test_api_importable():
    """
    Check that all submodules listed higher up in this file can be imported
    Note that if a PRIVATE_BUT_PRESENT_MODULES entry goes missing, it may
    simply need to be removed from the list (deprecation may or may not be
    needed - apply common sense).
    """

    def check_importable(module_name):
        try:
            importlib.import_module(module_name)
        except (ImportError, AttributeError):
            return False
        return True
    module_names = []
    for module_name in PUBLIC_MODULES:
        if not check_importable(module_name):
            module_names.append(module_name)
    if module_names:
        raise AssertionError(f'Modules in the public API that cannot be imported: {module_names}')
    with warnings.catch_warnings(record=True):
        warnings.filterwarnings('always', category=DeprecationWarning)
        warnings.filterwarnings('always', category=ImportWarning)
        for module_name in PRIVATE_BUT_PRESENT_MODULES:
            if not check_importable(module_name):
                module_names.append(module_name)
    if module_names:
        raise AssertionError(f'Modules that are not really public but looked public and can not be imported: {module_names}')