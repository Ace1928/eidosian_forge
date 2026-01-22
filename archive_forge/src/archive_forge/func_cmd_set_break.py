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
def cmd_set_break(self, py_db, cmd_id, seq, text):
    suspend_policy = u'NONE'
    is_logpoint = False
    hit_condition = None
    if py_db._set_breakpoints_with_id:
        try:
            try:
                breakpoint_id, btype, filename, line, func_name, condition, expression, hit_condition, is_logpoint, suspend_policy = text.split(u'\t', 9)
            except ValueError:
                breakpoint_id, btype, filename, line, func_name, condition, expression, hit_condition, is_logpoint = text.split(u'\t', 8)
            is_logpoint = is_logpoint == u'True'
        except ValueError:
            breakpoint_id, btype, filename, line, func_name, condition, expression = text.split(u'\t', 6)
        breakpoint_id = int(breakpoint_id)
        line = int(line)
        condition = condition.replace(u'@_@NEW_LINE_CHAR@_@', u'\n').replace(u'@_@TAB_CHAR@_@', u'\t').strip()
        expression = expression.replace(u'@_@NEW_LINE_CHAR@_@', u'\n').replace(u'@_@TAB_CHAR@_@', u'\t').strip()
    else:
        btype, filename, line, func_name, suspend_policy, condition, expression = text.split(u'\t', 6)
        breakpoint_id = line = int(line)
        condition = condition.replace(u'@_@NEW_LINE_CHAR@_@', u'\n').replace(u'@_@TAB_CHAR@_@', u'\t').strip()
        expression = expression.replace(u'@_@NEW_LINE_CHAR@_@', u'\n').replace(u'@_@TAB_CHAR@_@', u'\t').strip()
    if condition is not None and (len(condition) <= 0 or condition == u'None'):
        condition = None
    if expression is not None and (len(expression) <= 0 or expression == u'None'):
        expression = None
    if hit_condition is not None and (len(hit_condition) <= 0 or hit_condition == u'None'):
        hit_condition = None

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
    result = self.api.add_breakpoint(py_db, self.api.filename_to_str(filename), btype, breakpoint_id, line, condition, func_name, expression, suspend_policy, hit_condition, is_logpoint, on_changed_breakpoint_state=on_changed_breakpoint_state)
    on_changed_breakpoint_state(breakpoint_id, result)