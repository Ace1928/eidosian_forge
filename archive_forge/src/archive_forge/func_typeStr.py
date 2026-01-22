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
def typeStr(obj):
    """Create a more useful type string by making <instance> types report their class."""
    typ = type(obj)
    if typ == getattr(types, 'InstanceType', None):
        return '<instance of %s>' % obj.__class__.__name__
    else:
        return str(typ)