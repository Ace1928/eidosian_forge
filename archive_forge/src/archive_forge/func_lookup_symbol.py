from pyparsing import *
from sys import stdin, argv, exit
def lookup_symbol(self, sname, skind=list(SharedData.KINDS.keys()), stype=list(SharedData.TYPES.keys())):
    """Searches for symbol, from the end to the begining.
           Returns symbol index or None
           sname - symbol name
           skind - symbol kind (one kind, list of kinds, or None) deafult: any kind
           stype - symbol type (or None) default: any type
        """
    skind = skind if isinstance(skind, list) else [skind]
    stype = stype if isinstance(stype, list) else [stype]
    for i, sym in [[x, self.table[x]] for x in range(len(self.table) - 1, SharedData.LAST_WORKING_REGISTER, -1)]:
        if sym.name == sname and sym.kind in skind and (sym.type in stype):
            return i
    return None