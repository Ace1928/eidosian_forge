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
def execute_console_command(frame, thread_id, frame_id, line, buffer_output=True):
    """fetch an interactive console instance from the cache and
    push the received command to the console.

    create and return an instance of console_message
    """
    console_message = ConsoleMessage()
    interpreter = get_interactive_console(thread_id, frame_id, frame, console_message)
    more, output_messages, error_messages = interpreter.push(line, frame, buffer_output)
    console_message.update_more(more)
    for message in output_messages:
        console_message.add_console_message(CONSOLE_OUTPUT, message)
    for message in error_messages:
        console_message.add_console_message(CONSOLE_ERROR, message)
    return console_message