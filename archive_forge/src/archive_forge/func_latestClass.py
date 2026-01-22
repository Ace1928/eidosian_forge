import linecache
import sys
import time
import types
from importlib import reload
from types import ModuleType
from typing import Dict
from twisted.python import log, reflect
def latestClass(oldClass):
    """
    Get the latest version of a class.
    """
    module = reflect.namedModule(oldClass.__module__)
    newClass = getattr(module, oldClass.__name__)
    newBases = [latestClass(base) for base in newClass.__bases__]
    if newClass.__module__ == 'builtins':
        return newClass
    try:
        newClass.__bases__ = tuple(newBases)
        return newClass
    except TypeError:
        ctor = type(newClass)
        return ctor(newClass.__name__, tuple(newBases), dict(newClass.__dict__))