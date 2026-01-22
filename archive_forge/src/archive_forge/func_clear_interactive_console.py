import sys
import traceback
from _pydevd_bundle.pydevconsole_code import InteractiveConsole, _EvalAwaitInNewEventLoop
from _pydev_bundle import _pydev_completer
from _pydev_bundle.pydev_console_utils import BaseInterpreterInterface, BaseStdIn
from _pydev_bundle.pydev_imports import Exec
from _pydev_bundle.pydev_override import overrides
from _pydevd_bundle import pydevd_save_locals
from _pydevd_bundle.pydevd_io import IOBuf
from pydevd_tracing import get_exception_traceback_str
from _pydevd_bundle.pydevd_xml import make_valid_xml_value
import inspect
from _pydevd_bundle.pydevd_save_locals import update_globals_and_locals
def clear_interactive_console():
    InteractiveConsoleCache.thread_id = None
    InteractiveConsoleCache.frame_id = None
    InteractiveConsoleCache.interactive_console_instance = None