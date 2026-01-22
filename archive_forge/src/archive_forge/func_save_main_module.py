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
def save_main_module(file, module_name):
    m = sys.modules[module_name] = sys.modules['__main__']
    m.__name__ = module_name
    loader = m.__loader__ if hasattr(m, '__loader__') else None
    spec = spec_from_file_location('__main__', file, loader=loader)
    m = module_from_spec(spec)
    sys.modules['__main__'] = m
    return m