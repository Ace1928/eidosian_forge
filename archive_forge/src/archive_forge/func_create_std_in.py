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
def create_std_in(self, debugger=None, original_std_in=None):
    if debugger is None:
        return StdIn(self, self.host, self.client_port, original_stdin=original_std_in)
    else:
        return DebugConsoleStdIn(py_db=debugger, original_stdin=original_std_in)