import _symtable
from _symtable import (USE, DEF_GLOBAL, DEF_NONLOCAL, DEF_LOCAL, DEF_PARAM,
import weakref
def is_optimized(self):
    """Return *True* if the locals in the table
        are optimizable.
        """
    return bool(self._table.type == _symtable.TYPE_FUNCTION)