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
def printTrace(msg='', indent=4, prefix='|'):
    """Print an error message followed by an indented stack trace"""
    trace = backtrace(1)
    print('[%s]  %s\n' % (time.strftime('%H:%M:%S'), msg))
    print(' ' * indent + prefix + '=' * 30 + '>>')
    for line in trace.split('\n'):
        print(' ' * indent + prefix + ' ' + line)
    print(' ' * indent + prefix + '=' * 30 + '<<')