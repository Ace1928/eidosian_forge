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
def cmd_pydevd_json_config(self, py_db, cmd_id, seq, text):
    msg = json.loads(text.strip())
    if 'skip_suspend_on_breakpoint_exception' in msg:
        py_db.skip_suspend_on_breakpoint_exception = tuple((get_exception_class(x) for x in msg['skip_suspend_on_breakpoint_exception']))
    if 'skip_print_breakpoint_exception' in msg:
        py_db.skip_print_breakpoint_exception = tuple((get_exception_class(x) for x in msg['skip_print_breakpoint_exception']))
    if 'multi_threads_single_notification' in msg:
        py_db.multi_threads_single_notification = msg['multi_threads_single_notification']