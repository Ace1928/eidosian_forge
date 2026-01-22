import itertools
import json
import linecache
import os
import platform
import sys
from functools import partial
import pydevd_file_utils
from _pydev_bundle import pydev_log
from _pydevd_bundle._debug_adapter import pydevd_base_schema, pydevd_schema
from _pydevd_bundle._debug_adapter.pydevd_schema import (
from _pydevd_bundle.pydevd_api import PyDevdAPI
from _pydevd_bundle.pydevd_breakpoints import get_exception_class, FunctionBreakpoint
from _pydevd_bundle.pydevd_comm_constants import (
from _pydevd_bundle.pydevd_filtering import ExcludeFilter
from _pydevd_bundle.pydevd_json_debug_options import _extract_debug_options, DebugOptions
from _pydevd_bundle.pydevd_net_command import NetCommand
from _pydevd_bundle.pydevd_utils import convert_dap_log_message_to_expression, ScopeRequest
from _pydevd_bundle.pydevd_constants import (PY_IMPL_NAME, DebugInfoHolder, PY_VERSION_STR,
from _pydevd_bundle.pydevd_trace_dispatch import USING_CYTHON
from _pydevd_frame_eval.pydevd_frame_eval_main import USING_FRAME_EVAL
from _pydevd_bundle.pydevd_comm import internal_get_step_in_targets_json
from _pydevd_bundle.pydevd_additional_thread_info import set_additional_thread_info
from _pydevd_bundle.pydevd_thread_lifecycle import pydevd_find_thread_by_id
def on_setexceptionbreakpoints_request(self, py_db, request):
    """
        :param SetExceptionBreakpointsRequest request:
        """
    arguments = request.arguments
    filters = arguments.filters
    exception_options = arguments.exceptionOptions
    self.api.remove_all_exception_breakpoints(py_db)
    condition = None
    expression = None
    notify_on_first_raise_only = False
    ignore_libraries = 1 if py_db.get_use_libraries_filter() else 0
    if exception_options:
        break_raised = False
        break_uncaught = False
        for option in exception_options:
            option = ExceptionOptions(**option)
            if not option.path:
                continue
            notify_on_handled_exceptions = 1 if option.breakMode == 'always' else 0
            notify_on_unhandled_exceptions = 1 if option.breakMode == 'unhandled' else 0
            notify_on_user_unhandled_exceptions = 1 if option.breakMode == 'userUnhandled' else 0
            exception_paths = option.path
            break_raised |= notify_on_handled_exceptions
            break_uncaught |= notify_on_unhandled_exceptions
            exception_names = []
            if len(exception_paths) == 0:
                continue
            elif len(exception_paths) == 1:
                if 'Python Exceptions' in exception_paths[0]['names']:
                    exception_names = ['BaseException']
            else:
                path_iterator = iter(exception_paths)
                if 'Python Exceptions' in next(path_iterator)['names']:
                    for path in path_iterator:
                        for ex_name in path['names']:
                            exception_names.append(ex_name)
            for exception_name in exception_names:
                self.api.add_python_exception_breakpoint(py_db, exception_name, condition, expression, notify_on_handled_exceptions, notify_on_unhandled_exceptions, notify_on_user_unhandled_exceptions, notify_on_first_raise_only, ignore_libraries)
    else:
        break_raised = 'raised' in filters
        break_uncaught = 'uncaught' in filters
        break_user = 'userUnhandled' in filters
        if break_raised or break_uncaught or break_user:
            notify_on_handled_exceptions = 1 if break_raised else 0
            notify_on_unhandled_exceptions = 1 if break_uncaught else 0
            notify_on_user_unhandled_exceptions = 1 if break_user else 0
            exception = 'BaseException'
            self.api.add_python_exception_breakpoint(py_db, exception, condition, expression, notify_on_handled_exceptions, notify_on_unhandled_exceptions, notify_on_user_unhandled_exceptions, notify_on_first_raise_only, ignore_libraries)
    if break_raised:
        btype = None
        if self._options.django_debug:
            btype = 'django'
        elif self._options.flask_debug:
            btype = 'jinja2'
        if btype:
            self.api.add_plugins_exception_breakpoint(py_db, btype, 'BaseException')
    set_breakpoints_response = pydevd_base_schema.build_response(request)
    return NetCommand(CMD_RETURN, 0, set_breakpoints_response, is_json=True)