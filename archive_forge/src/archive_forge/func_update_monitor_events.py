from collections import namedtuple
import dis
import os
import re
import sys
from _pydev_bundle._pydev_saved_modules import threading
from types import CodeType, FrameType
from typing import Dict, Optional, Tuple, Any
from os.path import basename, splitext
from _pydev_bundle import pydev_log
from _pydevd_bundle import pydevd_dont_trace
from _pydevd_bundle.pydevd_constants import (GlobalDebuggerHolder, ForkSafeLock,
from pydevd_file_utils import (NORM_PATHS_AND_BASE_CONTAINER,
from _pydevd_bundle.pydevd_trace_dispatch import should_stop_on_exception, handle_exception
from _pydevd_bundle.pydevd_constants import EXCEPTION_TYPE_HANDLED
from _pydevd_bundle.pydevd_trace_dispatch import is_unhandled_exception
from _pydevd_bundle.pydevd_breakpoints import stop_on_unhandled_exception
from _pydevd_bundle.pydevd_utils import get_clsname_for_code
from _pydevd_bundle.pydevd_additional_thread_info import set_additional_thread_info, any_thread_stepping, PyDBAdditionalThreadInfo
def update_monitor_events(suspend_requested: Optional[bool]=None) -> None:
    """
    This should be called when breakpoints change.

    :param suspend: means the user requested threads to be suspended
    """
    if monitor.get_tool(monitor.DEBUGGER_ID) != 'pydevd':
        return
    py_db = GlobalDebuggerHolder.global_dbg
    if py_db is None:
        return
    if suspend_requested is None:
        suspend_requested = False
        for t in threading.enumerate():
            if getattr(t, 'pydev_do_not_trace', False):
                continue
            try:
                additional_info = t.additional_info
                if additional_info is None:
                    continue
            except AttributeError:
                continue
            if additional_info.pydev_step_cmd != -1 or additional_info.pydev_state == 2:
                suspend_requested = True
                break
    required_events = 0
    has_caught_exception_breakpoint_in_pydb = py_db.break_on_caught_exceptions or py_db.break_on_user_uncaught_exceptions or py_db.has_plugin_exception_breaks
    break_on_uncaught_exceptions = py_db.break_on_uncaught_exceptions
    if has_caught_exception_breakpoint_in_pydb:
        required_events |= monitor.events.RAISE | monitor.events.PY_UNWIND
        monitor.register_callback(DEBUGGER_ID, monitor.events.RAISE, _raise_event)
        monitor.register_callback(DEBUGGER_ID, monitor.events.PY_UNWIND, _unwind_event)
    elif break_on_uncaught_exceptions:
        required_events |= monitor.events.PY_UNWIND
        monitor.register_callback(DEBUGGER_ID, monitor.events.PY_UNWIND, _unwind_event)
    else:
        monitor.register_callback(DEBUGGER_ID, monitor.events.RAISE, None)
        monitor.register_callback(DEBUGGER_ID, monitor.events.PY_UNWIND, None)
    has_breaks = py_db.has_plugin_line_breaks
    if not has_breaks:
        if py_db.function_breakpoint_name_to_breakpoint:
            has_breaks = True
        else:
            file_to_line_to_breakpoints = py_db.breakpoints
            for line_to_breakpoints in file_to_line_to_breakpoints.values():
                if line_to_breakpoints:
                    has_breaks = True
                    break
    if has_breaks or suspend_requested:
        required_events |= monitor.events.PY_START | monitor.events.PY_RESUME
        monitor.register_callback(DEBUGGER_ID, monitor.events.PY_START, _start_method_event)
        monitor.register_callback(DEBUGGER_ID, monitor.events.LINE, _line_event)
        monitor.register_callback(DEBUGGER_ID, monitor.events.JUMP, _jump_event)
        monitor.register_callback(DEBUGGER_ID, monitor.events.PY_RETURN, _return_event)
    else:
        monitor.register_callback(DEBUGGER_ID, monitor.events.PY_START, None)
        monitor.register_callback(DEBUGGER_ID, monitor.events.PY_RESUME, None)
        monitor.register_callback(DEBUGGER_ID, monitor.events.LINE, None)
        monitor.register_callback(DEBUGGER_ID, monitor.events.JUMP, None)
        monitor.register_callback(DEBUGGER_ID, monitor.events.PY_RETURN, None)
    monitor.set_events(DEBUGGER_ID, required_events)