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
def on_changed_breakpoint_state(breakpoint_id, add_breakpoint_result):
    error_code = add_breakpoint_result.error_code
    translated_line = add_breakpoint_result.translated_line
    translated_filename = add_breakpoint_result.translated_filename
    msg = ''
    if error_code:
        if error_code == self.api.ADD_BREAKPOINT_FILE_NOT_FOUND:
            msg = 'pydev debugger: Trying to add breakpoint to file that does not exist: %s (will have no effect).\n' % (translated_filename,)
        elif error_code == self.api.ADD_BREAKPOINT_FILE_EXCLUDED_BY_FILTERS:
            msg = 'pydev debugger: Trying to add breakpoint to file that is excluded by filters: %s (will have no effect).\n' % (translated_filename,)
        elif error_code == self.api.ADD_BREAKPOINT_LAZY_VALIDATION:
            msg = ''
        elif error_code == self.api.ADD_BREAKPOINT_INVALID_LINE:
            msg = 'pydev debugger: Trying to add breakpoint to line (%s) that is not valid in: %s.\n' % (translated_line, translated_filename)
        else:
            msg = 'pydev debugger: Breakpoint not validated (reason unknown -- please report as error): %s (%s).\n' % (translated_filename, translated_line)
    elif add_breakpoint_result.original_line != translated_line:
        msg = 'pydev debugger (info): Breakpoint in line: %s moved to line: %s (in %s).\n' % (add_breakpoint_result.original_line, translated_line, translated_filename)
    if msg:
        py_db.writer.add_command(py_db.cmd_factory.make_warning_message(msg))