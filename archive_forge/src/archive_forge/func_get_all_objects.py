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
def get_all_objects():
    """Return a list of all live Python objects (excluding int and long), not including the list itself."""
    gc.collect()
    gcl = gc.get_objects()
    olist = {}
    _getr(gcl, olist)
    del olist[id(olist)]
    del olist[id(gcl)]
    del olist[id(sys._getframe())]
    return olist