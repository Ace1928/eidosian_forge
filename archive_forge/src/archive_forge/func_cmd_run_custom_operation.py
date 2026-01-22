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
def cmd_run_custom_operation(self, py_db, cmd_id, seq, text):
    if text != '':
        try:
            location, custom = text.split('||', 1)
        except:
            sys.stderr.write('Custom operation now needs a || separator. Found: %s\n' % (text,))
            raise
        thread_id, frame_id, scopeattrs = location.split('\t', 2)
        if scopeattrs.find('\t') != -1:
            scope, attrs = scopeattrs.split('\t', 1)
        else:
            scope, attrs = (scopeattrs, None)
        style, encoded_code_or_file, fnname = custom.split('\t', 3)
        int_cmd = InternalRunCustomOperation(seq, thread_id, frame_id, scope, attrs, style, encoded_code_or_file, fnname)
        py_db.post_internal_command(int_cmd, thread_id)