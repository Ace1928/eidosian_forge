import _symtable
from _symtable import (USE, DEF_GLOBAL, DEF_NONLOCAL, DEF_LOCAL, DEF_PARAM,
import weakref
def get_symbols(self):
    """Return a list of *Symbol* instances for
        names in the table.
        """
    return [self.lookup(ident) for ident in self.get_identifiers()]