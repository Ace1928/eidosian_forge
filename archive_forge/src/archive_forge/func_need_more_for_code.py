import os
import sys
import traceback
from _pydev_bundle.pydev_imports import xmlrpclib, _queue, Exec
from  _pydev_bundle._pydev_calltip_util import get_description
from _pydevd_bundle import pydevd_vars
from _pydevd_bundle import pydevd_xml
from _pydevd_bundle.pydevd_constants import (IS_JYTHON, NEXT_VALUE_SEPARATOR, get_global_debugger,
from contextlib import contextmanager
from _pydev_bundle import pydev_log
from _pydevd_bundle.pydevd_utils import interrupt_main_thread
from io import StringIO
def need_more_for_code(self, source):
    if source.endswith('\\'):
        return True
    if hasattr(self.interpreter, 'is_complete'):
        return not self.interpreter.is_complete(source)
    try:
        symbol = 'single'
        code = self.interpreter.compile(source, '<input>', symbol)
    except (OverflowError, SyntaxError, ValueError):
        return False
    if code is None:
        return True
    return False