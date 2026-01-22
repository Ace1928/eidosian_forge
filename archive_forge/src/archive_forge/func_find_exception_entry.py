from collections import namedtuple, OrderedDict
import dis
import inspect
import itertools
from types import CodeType, ModuleType
from numba.core import errors, utils, serialize
from numba.core.utils import PYVERSION
def find_exception_entry(self, offset):
    """
        Returns the exception entry for the given instruction offset
        """
    candidates = []
    for ent in self.exception_entries:
        if ent.start <= offset < ent.end:
            candidates.append((ent.depth, ent))
    if candidates:
        ent = max(candidates)[1]
        return ent