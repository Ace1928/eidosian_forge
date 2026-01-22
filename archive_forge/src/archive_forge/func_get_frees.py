import _symtable
from _symtable import (USE, DEF_GLOBAL, DEF_NONLOCAL, DEF_LOCAL, DEF_PARAM,
import weakref
def get_frees(self):
    """Return a tuple of free variables in the function.
        """
    if self.__frees is None:
        is_free = lambda x: x >> SCOPE_OFF & SCOPE_MASK == FREE
        self.__frees = self.__idents_matching(is_free)
    return self.__frees