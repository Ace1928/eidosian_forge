from __future__ import print_function
import sys
import os
import types
import traceback
from abc import abstractmethod
def dbg(val=None):
    global _debug
    if _debug is None:
        _debugx = os.environ.get('YAMLDEBUG')
        if _debugx is None:
            _debug = 0
        else:
            _debug = int(_debugx)
    if val is None:
        return _debug
    return _debug & val