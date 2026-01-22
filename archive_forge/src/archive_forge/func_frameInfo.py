from __future__ import print_function
import contextlib
import cProfile
import gc
import inspect
import os
import re
import sys
import threading
import time
import traceback
import types
import warnings
import weakref
from time import perf_counter
from numpy import ndarray
from .Qt import QT_LIB, QtCore
from .util import cprint
from .util.mutex import Mutex
def frameInfo(self, fr):
    filename = fr.f_code.co_filename
    funcname = fr.f_code.co_name
    lineno = fr.f_lineno
    callfr = sys._getframe(3)
    callline = '%s %d' % (callfr.f_code.co_name, callfr.f_lineno)
    args, _, _, value_dict = inspect.getargvalues(fr)
    if len(args) and args[0] == 'self':
        instance = value_dict.get('self', None)
        if instance is not None:
            cls = getattr(instance, '__class__', None)
            if cls is not None:
                funcname = cls.__name__ + '.' + funcname
    return '%s: %s %s: %s' % (callline, filename, lineno, funcname)