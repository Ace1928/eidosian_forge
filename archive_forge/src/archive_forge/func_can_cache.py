import weakref
import importlib
from numba import _dynfunc
def can_cache(self):
    is_dyn = '__name__' not in self.globals
    return not is_dyn