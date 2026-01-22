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
def cmd_evaluate_console_expression(self, py_db, cmd_id, seq, text):
    if text != '':
        thread_id, frame_id, console_command = text.split('\t', 2)
        console_command, line = console_command.split('\t')
        if console_command == 'EVALUATE':
            int_cmd = InternalEvaluateConsoleExpression(seq, thread_id, frame_id, line, buffer_output=True)
        elif console_command == 'EVALUATE_UNBUFFERED':
            int_cmd = InternalEvaluateConsoleExpression(seq, thread_id, frame_id, line, buffer_output=False)
        elif console_command == 'GET_COMPLETIONS':
            int_cmd = InternalConsoleGetCompletions(seq, thread_id, frame_id, line)
        else:
            raise ValueError('Unrecognized command: %s' % (console_command,))
        py_db.post_internal_command(int_cmd, thread_id)