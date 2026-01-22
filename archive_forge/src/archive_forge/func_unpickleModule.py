import copy
import copyreg as copy_reg
import inspect
import pickle
import types
from io import StringIO as _cStringIO
from typing import Dict
from twisted.python import log, reflect
from twisted.python.compat import _PYPY
def unpickleModule(name):
    """support function for copy_reg to unpickle module refs"""
    if name in oldModules:
        log.msg('Module has moved: %s' % name)
        name = oldModules[name]
        log.msg(name)
    return __import__(name, {}, {}, 'x')