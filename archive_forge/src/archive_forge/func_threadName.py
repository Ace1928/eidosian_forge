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
def threadName(threadId=None):
    """Return a string name for a thread id.

    If *threadId* is None, then the current thread's id is used.

    This attempts to look up thread names either from `threading._active`, or from
    QThread._names. However, note that the latter does not exist by default; rather
    you must manually add id:name pairs to a dictionary there::

        # for python threads:
        t1 = threading.Thread(name="mythread")

        # for Qt threads:
        class Thread(Qt.QThread):
            def __init__(self, name):
                self._threadname = name
                if not hasattr(Qt.QThread, '_names'):
                    Qt.QThread._names = {}
                Qt.QThread.__init__(self, *args, **kwds)
            def run(self):
                Qt.QThread._names[threading.current_thread().ident] = self._threadname
    """
    if threadId is None:
        threadId = threading.current_thread().ident
    try:
        name = threading._active.get(threadId, None)
    except Exception:
        name = None
    if name is None:
        try:
            name = QtCore.QThread._names.get(threadId)
        except Exception:
            name = None
    if name is None:
        name = '???'
    return name