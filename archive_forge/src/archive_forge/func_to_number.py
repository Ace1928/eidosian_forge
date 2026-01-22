from __future__ import nested_scopes
import traceback
import warnings
from _pydev_bundle import pydev_log
from _pydev_bundle._pydev_saved_modules import thread, threading
from _pydev_bundle import _pydev_saved_modules
import signal
import os
import ctypes
from importlib import import_module
from importlib.util import module_from_spec, spec_from_file_location
from urllib.parse import quote  # @UnresolvedImport
import time
import inspect
import sys
from _pydevd_bundle.pydevd_constants import USE_CUSTOM_SYS_CURRENT_FRAMES, IS_PYPY, SUPPORT_GEVENT, \
def to_number(x):
    if is_string(x):
        try:
            n = float(x)
            return n
        except ValueError:
            pass
        l = x.find('(')
        if l != -1:
            y = x[0:l - 1]
            try:
                n = float(y)
                return n
            except ValueError:
                pass
    return None