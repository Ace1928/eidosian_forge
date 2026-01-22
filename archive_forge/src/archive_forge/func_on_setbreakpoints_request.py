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
def on_setbreakpoints_request(self, py_db, request):
    """
        :param SetBreakpointsRequest request:
        """
    response = self._verify_launch_or_attach_done(request)
    if response is not None:
        return response
    arguments = request.arguments
    filename = self.api.filename_to_str(arguments.source.path)
    func_name = 'None'
    self.api.remove_all_breakpoints(py_db, filename)
    btype = 'python-line'
    suspend_policy = 'ALL'
    if not filename.lower().endswith('.py'):
        if self._options.django_debug:
            btype = 'django-line'
        elif self._options.flask_debug:
            btype = 'jinja2-line'
    breakpoints_set = []
    for source_breakpoint in arguments.breakpoints:
        source_breakpoint = SourceBreakpoint(**source_breakpoint)
        line = source_breakpoint.line
        condition = source_breakpoint.condition
        breakpoint_id = self._next_breakpoint_id()
        hit_condition = self._get_hit_condition_expression(source_breakpoint.hitCondition)
        log_message = source_breakpoint.logMessage
        if not log_message:
            is_logpoint = None
            expression = None
        else:
            is_logpoint = True
            expression = convert_dap_log_message_to_expression(log_message)
        on_changed_breakpoint_state = partial(self._on_changed_breakpoint_state, py_db, arguments.source)
        result = self.api.add_breakpoint(py_db, filename, btype, breakpoint_id, line, condition, func_name, expression, suspend_policy, hit_condition, is_logpoint, adjust_line=True, on_changed_breakpoint_state=on_changed_breakpoint_state)
        bp = self._create_breakpoint_from_add_breakpoint_result(py_db, arguments.source, breakpoint_id, result)
        breakpoints_set.append(bp)
    body = {'breakpoints': breakpoints_set}
    set_breakpoints_response = pydevd_base_schema.build_response(request, kwargs={'body': body})
    return NetCommand(CMD_RETURN, 0, set_breakpoints_response, is_json=True)