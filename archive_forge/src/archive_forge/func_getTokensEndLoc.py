import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
def getTokensEndLoc():
    """Method to be called from within a parse action to determine the end
       location of the parsed tokens."""
    import inspect
    fstack = inspect.stack()
    try:
        for f in fstack[2:]:
            if f[3] == '_parseNoCache':
                endloc = f[0].f_locals['loc']
                return endloc
        else:
            raise ParseFatalException('incorrect usage of getTokensEndLoc - may only be called from within a parse action')
    finally:
        del fstack