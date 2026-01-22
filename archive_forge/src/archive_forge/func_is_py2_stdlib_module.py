from __future__ import absolute_import, division, print_function
import sys
import logging
import contextlib
import copy
import os
from future.utils import PY2, PY3
def is_py2_stdlib_module(m):
    """
    Tries to infer whether the module m is from the Python 2 standard library.
    This may not be reliable on all systems.
    """
    if PY3:
        return False
    if not 'stdlib_path' in is_py2_stdlib_module.__dict__:
        stdlib_files = [contextlib.__file__, os.__file__, copy.__file__]
        stdlib_paths = [os.path.split(f)[0] for f in stdlib_files]
        if not len(set(stdlib_paths)) == 1:
            flog.warn('Multiple locations found for the Python standard library: %s' % stdlib_paths)
        is_py2_stdlib_module.stdlib_path = stdlib_paths[0]
    if m.__name__ in sys.builtin_module_names:
        return True
    if hasattr(m, '__file__'):
        modpath = os.path.split(m.__file__)
        if modpath[0].startswith(is_py2_stdlib_module.stdlib_path) and 'site-packages' not in modpath[0]:
            return True
    return False