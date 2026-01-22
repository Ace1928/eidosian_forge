import linecache
import re
from inspect import (getblock, getfile, getmodule, getsourcefile, indentsize,
from tokenize import TokenError
from ._dill import IS_IPYTHON
def isdynamic(obj):
    """check if object was built in the interpreter"""
    try:
        file = getfile(obj)
    except TypeError:
        file = None
    if file == '<stdin>' and isfrommain(obj):
        return True
    return False