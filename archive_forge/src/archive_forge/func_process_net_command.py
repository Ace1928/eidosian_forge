import json
import os
import sys
import traceback
from _pydev_bundle import pydev_log
from _pydev_bundle.pydev_log import exception as pydev_log_exception
from _pydevd_bundle import pydevd_traceproperty, pydevd_dont_trace, pydevd_utils
from _pydevd_bundle.pydevd_additional_thread_info import set_additional_thread_info
from _pydevd_bundle.pydevd_breakpoints import get_exception_class
from _pydevd_bundle.pydevd_comm import (
from _pydevd_bundle.pydevd_constants import NEXT_VALUE_SEPARATOR, IS_WINDOWS, NULL
from _pydevd_bundle.pydevd_comm_constants import ID_TO_MEANING, CMD_EXEC_EXPRESSION, CMD_AUTHENTICATE
from _pydevd_bundle.pydevd_api import PyDevdAPI
from io import StringIO
from _pydevd_bundle.pydevd_net_command import NetCommand
from _pydevd_bundle.pydevd_thread_lifecycle import pydevd_find_thread_by_id
import pydevd_file_utils
def process_net_command(self, py_db, cmd_id, seq, text):
    """Processes a command received from the Java side

        @param cmd_id: the id of the command
        @param seq: the sequence of the command
        @param text: the text received in the command
        """
    if cmd_id != CMD_AUTHENTICATE and (not py_db.authentication.is_authenticated()):
        cmd = py_db.cmd_factory.make_error_message(seq, 'Client not authenticated.')
        py_db.writer.add_command(cmd)
        return
    meaning = ID_TO_MEANING[str(cmd_id)]
    method_name = meaning.lower()
    on_command = getattr(self, method_name.lower(), None)
    if on_command is None:
        cmd = py_db.cmd_factory.make_error_message(seq, 'unexpected command ' + str(cmd_id))
        py_db.writer.add_command(cmd)
        return
    lock = py_db._main_lock
    if method_name == 'cmd_thread_dump_to_stderr':
        lock = NULL
    with lock:
        try:
            cmd = on_command(py_db, cmd_id, seq, text)
            if cmd is not None:
                py_db.writer.add_command(cmd)
        except:
            if traceback is not None and sys is not None and (pydev_log_exception is not None):
                pydev_log_exception()
                stream = StringIO()
                traceback.print_exc(file=stream)
                cmd = py_db.cmd_factory.make_error_message(seq, 'Unexpected exception in process_net_command.\nInitial params: %s. Exception: %s' % ((cmd_id, seq, text), stream.getvalue()))
                if cmd is not None:
                    py_db.writer.add_command(cmd)