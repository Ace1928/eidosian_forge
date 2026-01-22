import pkgutil
import types
import importlib
import warnings
from importlib import import_module
import pytest
import scipy
def test_all_modules_are_expected_2():
    """
    Method checking all objects. The pkgutil-based method in
    `test_all_modules_are_expected` does not catch imports into a namespace,
    only filenames.
    """

    def find_unexpected_members(mod_name):
        members = []
        module = importlib.import_module(mod_name)
        if hasattr(module, '__all__'):
            objnames = module.__all__
        else:
            objnames = dir(module)
        for objname in objnames:
            if not objname.startswith('_'):
                fullobjname = mod_name + '.' + objname
                if isinstance(getattr(module, objname), types.ModuleType):
                    if is_unexpected(fullobjname) and fullobjname not in SKIP_LIST_2:
                        members.append(fullobjname)
        return members
    unexpected_members = find_unexpected_members('scipy')
    for modname in PUBLIC_MODULES:
        unexpected_members.extend(find_unexpected_members(modname))
    if unexpected_members:
        raise AssertionError(f'Found unexpected object(s) that look like modules: {unexpected_members}')