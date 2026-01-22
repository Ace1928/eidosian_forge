from pyparsing import *
from sys import stdin, argv, exit
def insert_id(self, sname, skind, skinds, stype):
    """Inserts a new identifier at the end of the symbol table, if possible.
           Returns symbol index, or raises an exception if the symbol alredy exists
           sname   - symbol name
           skind   - symbol kind
           skinds  - symbol kinds to check for
           stype   - symbol type
        """
    index = self.lookup_symbol(sname, skinds)
    if index == None:
        index = self.insert_symbol(sname, skind, stype)
        return index
    else:
        raise SemanticException("Redefinition of '%s'" % sname)