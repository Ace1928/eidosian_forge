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
def convert_dap_log_message_to_expression(log_message):
    try:
        expression, expression_vars = _extract_expression_list(log_message)
    except SyntaxError:
        return repr('Unbalanced braces in: %s' % log_message)
    if not expression_vars:
        return repr(expression)
    return repr(expression) + ' % (' + ', '.join((str(x) for x in expression_vars)) + ',)'