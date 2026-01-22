import linecache
import sys
import time
import types
from importlib import reload
from types import ModuleType
from typing import Dict
from twisted.python import log, reflect
def latestVersionOf(self, anObject):
    """
        Get the latest version of an object.

        This can handle just about anything callable; instances, functions,
        methods, and classes.
        """
    t = type(anObject)
    if t == types.FunctionType:
        return latestFunction(anObject)
    elif t == types.MethodType:
        if anObject.__self__ is None:
            return getattr(anObject.im_class, anObject.__name__)
        else:
            return getattr(anObject.__self__, anObject.__name__)
    else:
        log.msg('warning returning anObject!')
        return anObject