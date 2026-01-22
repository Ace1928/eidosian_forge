from pyparsing import *
from sys import stdin, argv, exit
def insert_symbol(self, sname, skind, stype):
    """Inserts new symbol at the end of the symbol table.
           Returns symbol index
           sname - symbol name
           skind - symbol kind
           stype - symbol type
        """
    self.table.append(SymbolTableEntry(sname, skind, stype))
    self.table_len = len(self.table)
    return self.table_len - 1