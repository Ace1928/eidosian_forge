import sys
import marshal
import contextlib
import dis
from . import _imp
from ._imp import find_module, PY_COMPILED, PY_FROZEN, PY_SOURCE
from .extern.packaging.version import Version
def is_present(self, paths=None):
    """Return true if dependency is present on 'paths'"""
    return self.get_version(paths) is not None