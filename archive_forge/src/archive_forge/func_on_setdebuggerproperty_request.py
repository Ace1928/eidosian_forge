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
def on_setdebuggerproperty_request(self, py_db, request):
    args = request.arguments
    if args.ideOS is not None:
        self.api.set_ide_os(args.ideOS)
    if args.dontTraceStartPatterns is not None and args.dontTraceEndPatterns is not None:
        start_patterns = tuple(args.dontTraceStartPatterns)
        end_patterns = tuple(args.dontTraceEndPatterns)
        self.api.set_dont_trace_start_end_patterns(py_db, start_patterns, end_patterns)
    if args.skipSuspendOnBreakpointException is not None:
        py_db.skip_suspend_on_breakpoint_exception = tuple((get_exception_class(x) for x in args.skipSuspendOnBreakpointException))
    if args.skipPrintBreakpointException is not None:
        py_db.skip_print_breakpoint_exception = tuple((get_exception_class(x) for x in args.skipPrintBreakpointException))
    if args.multiThreadsSingleNotification is not None:
        py_db.multi_threads_single_notification = args.multiThreadsSingleNotification
    response = pydevd_base_schema.build_response(request, kwargs={'body': {}})
    return NetCommand(CMD_RETURN, 0, response, is_json=True)