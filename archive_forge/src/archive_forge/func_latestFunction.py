import linecache
import sys
import time
import types
from importlib import reload
from types import ModuleType
from typing import Dict
from twisted.python import log, reflect
def latestFunction(oldFunc):
    """
    Get the latest version of a function.
    """
    dictID = id(oldFunc.__globals__)
    module = _modDictIDMap.get(dictID)
    if module is None:
        return oldFunc
    return getattr(module, oldFunc.__name__)