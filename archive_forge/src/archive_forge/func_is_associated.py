from .util import (
import sys
from functools import reduce
def is_associated(self):
    """:return: True if we are associated with a specific file already"""
    return self._rlist is not None