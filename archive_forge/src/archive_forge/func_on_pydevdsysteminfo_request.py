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
def on_pydevdsysteminfo_request(self, py_db, request):
    try:
        pid = os.getpid()
    except AttributeError:
        pid = None
    ppid = py_db.get_arg_ppid() or self.api.get_ppid()
    try:
        impl_desc = platform.python_implementation()
    except AttributeError:
        impl_desc = PY_IMPL_NAME
    py_info = pydevd_schema.PydevdPythonInfo(version=PY_VERSION_STR, implementation=pydevd_schema.PydevdPythonImplementationInfo(name=PY_IMPL_NAME, version=PY_IMPL_VERSION_STR, description=impl_desc))
    platform_info = pydevd_schema.PydevdPlatformInfo(name=sys.platform)
    process_info = pydevd_schema.PydevdProcessInfo(pid=pid, ppid=ppid, executable=sys.executable, bitness=64 if IS_64BIT_PROCESS else 32)
    pydevd_info = pydevd_schema.PydevdInfo(usingCython=USING_CYTHON, usingFrameEval=USING_FRAME_EVAL)
    body = {'python': py_info, 'platform': platform_info, 'process': process_info, 'pydevd': pydevd_info}
    response = pydevd_base_schema.build_response(request, kwargs={'body': body})
    return NetCommand(CMD_RETURN, 0, response, is_json=True)