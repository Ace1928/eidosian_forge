from pyparsing import *
from sys import stdin, argv, exit
def clear_symbols(self, index):
    """Clears all symbols begining with the index to the end of table"""
    try:
        del self.table[index:]
    except Exception:
        self.error()
    self.table_len = len(self.table)