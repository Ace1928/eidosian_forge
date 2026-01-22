import re
import sys
import subprocess
def parse_nm(nm_output):
    """Returns a tuple of lists: dlist for the list of data
symbols and flist for the list of function symbols.

dlist, flist = parse_nm(nm_output)"""
    data = DATA_RE.findall(nm_output)
    func = FUNC_RE.findall(nm_output)
    flist = []
    for sym in data:
        if sym in func and (sym[:2] == 'Py' or sym[:3] == '_Py' or sym[:4] == 'init'):
            flist.append(sym)
    dlist = []
    for sym in data:
        if sym not in flist and (sym[:2] == 'Py' or sym[:3] == '_Py'):
            dlist.append(sym)
    dlist.sort()
    flist.sort()
    return (dlist, flist)