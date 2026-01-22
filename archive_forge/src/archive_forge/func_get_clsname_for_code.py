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
def get_clsname_for_code(code, frame):
    clsname = None
    if len(code.co_varnames) > 0:
        first_arg_name = code.co_varnames[0]
        if first_arg_name in frame.f_locals:
            first_arg_obj = frame.f_locals[first_arg_name]
            if inspect.isclass(first_arg_obj):
                first_arg_class = first_arg_obj
            elif hasattr(first_arg_obj, '__class__'):
                first_arg_class = first_arg_obj.__class__
            else:
                first_arg_class = type(first_arg_obj)
            func_name = code.co_name
            if hasattr(first_arg_class, func_name):
                method = getattr(first_arg_class, func_name)
                func_code = None
                if hasattr(method, 'func_code'):
                    func_code = method.func_code
                elif hasattr(method, '__code__'):
                    func_code = method.__code__
                if func_code and func_code == code:
                    clsname = first_arg_class.__name__
    return clsname