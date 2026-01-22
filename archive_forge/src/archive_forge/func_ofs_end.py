from .util import (
import sys
from functools import reduce
def ofs_end(self):
    """:return: offset to one past the last available byte"""
    return self._region._b + self._ofs + self._size