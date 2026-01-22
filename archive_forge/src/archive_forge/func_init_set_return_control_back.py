from _pydev_bundle._pydev_saved_modules import thread, _code
from _pydevd_bundle.pydevd_constants import IS_JYTHON
from _pydevd_bundle.pydevconsole_code import InteractiveConsole
import os
import sys
from _pydev_bundle._pydev_saved_modules import threading
from _pydevd_bundle.pydevd_constants import INTERACTIVE_MODE_AVAILABLE
import traceback
from _pydev_bundle import pydev_log
from _pydevd_bundle import pydevd_save_locals
from _pydev_bundle.pydev_imports import Exec, _queue
import builtins as __builtin__
from _pydev_bundle.pydev_console_utils import BaseInterpreterInterface, BaseStdIn  # @UnusedImport
from _pydev_bundle.pydev_console_utils import CodeFragment
from _pydev_bundle.pydev_umd import runfile, _set_globals_function
def init_set_return_control_back(interpreter):
    from pydev_ipython.inputhook import set_return_control_callback

    def return_control():
        """ A function that the inputhooks can call (via inputhook.stdin_ready()) to find
            out if they should cede control and return """
        if _ProcessExecQueueHelper._debug_hook:
            _ProcessExecQueueHelper._return_control_osc = not _ProcessExecQueueHelper._return_control_osc
            if _ProcessExecQueueHelper._return_control_osc:
                return True
        if not interpreter.exec_queue.empty():
            return True
        return False
    set_return_control_callback(return_control)