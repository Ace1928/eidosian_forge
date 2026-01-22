import _symtable
from _symtable import (USE, DEF_GLOBAL, DEF_NONLOCAL, DEF_LOCAL, DEF_PARAM,
import weakref
def get_locals(self):
    """Return a tuple of locals in the function.
        """
    if self.__locals is None:
        locs = (LOCAL, CELL)
        test = lambda x: x >> SCOPE_OFF & SCOPE_MASK in locs
        self.__locals = self.__idents_matching(test)
    return self.__locals