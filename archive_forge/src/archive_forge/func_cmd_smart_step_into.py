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
def cmd_smart_step_into(self, py_db, cmd_id, seq, text):
    thread_id, line_or_bytecode_offset, func_name = text.split('\t', 2)
    if line_or_bytecode_offset.startswith('offset='):
        temp = line_or_bytecode_offset[len('offset='):]
        if ';' in temp:
            offset, child_offset = temp.split(';')
            offset = int(offset)
            child_offset = int(child_offset)
        else:
            child_offset = -1
            offset = int(temp)
        return self.api.request_smart_step_into(py_db, seq, thread_id, offset, child_offset)
    else:
        return self.api.request_smart_step_into_by_func_name(py_db, seq, thread_id, line_or_bytecode_offset, func_name)