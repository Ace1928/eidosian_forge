import functools
import inspect
import platform
import sys
import types
from importlib import import_module
from typing import List, TypeVar
import distutils.filelist
def get_unpatched_class(cls):
    """Protect against re-patching the distutils if reloaded

    Also ensures that no other distutils extension monkeypatched the distutils
    first.
    """
    external_bases = (cls for cls in _get_mro(cls) if not cls.__module__.startswith('setuptools'))
    base = next(external_bases)
    if not base.__module__.startswith('distutils'):
        msg = 'distutils has already been patched by %r' % cls
        raise AssertionError(msg)
    return base