import sys
import marshal
import contextlib
import dis
from . import _imp
from ._imp import find_module, PY_COMPILED, PY_FROZEN, PY_SOURCE
from .extern.packaging.version import Version
def version_ok(self, version):
    """Is 'version' sufficiently up-to-date?"""
    return self.attribute is None or self.format is None or (str(version) != 'unknown' and self.format(version) >= self.requested_version)