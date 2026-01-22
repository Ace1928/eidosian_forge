from .util import (
import sys
from functools import reduce
def num_open_files(self):
    """Amount of opened files in the system"""
    return reduce(lambda x, y: x + y, (1 for rlist in self._fdict.values() if len(rlist) > 0), 0)