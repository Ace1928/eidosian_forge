from __future__ import absolute_import
import itertools
import sys
from weakref import ref
@atexit.setter
def atexit(self, value):
    info = self._registry.get(self)
    if info:
        info.atexit = bool(value)