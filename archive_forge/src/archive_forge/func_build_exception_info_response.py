import linecache
import os
from _pydev_bundle.pydev_imports import _queue
from _pydev_bundle._pydev_saved_modules import time
from _pydev_bundle._pydev_saved_modules import threading
from _pydev_bundle._pydev_saved_modules import socket as socket_module
from _pydevd_bundle.pydevd_constants import (DebugInfoHolder, IS_WINDOWS, IS_JYTHON, IS_WASM,
from _pydev_bundle.pydev_override import overrides
import weakref
from _pydev_bundle._pydev_completer import extract_token_and_qualifier
from _pydevd_bundle._debug_adapter.pydevd_schema import VariablesResponseBody, \
from _pydevd_bundle._debug_adapter import pydevd_base_schema, pydevd_schema
from _pydevd_bundle.pydevd_net_command import NetCommand
from _pydevd_bundle.pydevd_xml import ExceptionOnEvaluate
from _pydevd_bundle.pydevd_constants import ForkSafeLock, NULL
from _pydevd_bundle.pydevd_daemon_thread import PyDBDaemonThread
from _pydevd_bundle.pydevd_thread_lifecycle import pydevd_find_thread_by_id, resume_threads
from _pydevd_bundle.pydevd_dont_trace_files import PYDEV_FILE
import dis
import pydevd_file_utils
import itertools
from urllib.parse import quote_plus, unquote_plus
import pydevconsole
from _pydevd_bundle import pydevd_vars, pydevd_io, pydevd_reload
from _pydevd_bundle import pydevd_bytecode_utils
from _pydevd_bundle import pydevd_xml
from _pydevd_bundle import pydevd_vm_type
import sys
import traceback
from _pydevd_bundle.pydevd_utils import quote_smart as quote, compare_object_attrs_key, \
from _pydev_bundle import pydev_log, fsnotify
from _pydev_bundle.pydev_log import exception as pydev_log_exception
from _pydev_bundle import _pydev_completer
from pydevd_tracing import get_exception_traceback_str
from _pydevd_bundle import pydevd_console
from _pydev_bundle.pydev_monkey import disable_trace_thread_modules, enable_trace_thread_modules
from io import StringIO
from _pydevd_bundle.pydevd_comm_constants import *  # @UnusedWildImport
def build_exception_info_response(dbg, thread_id, thread, request_seq, set_additional_thread_info, iter_visible_frames_info, max_frames):
    """
    :return ExceptionInfoResponse
    """
    additional_info = set_additional_thread_info(thread)
    topmost_frame = additional_info.get_topmost_frame(thread)
    current_paused_frame_name = ''
    source_path = ''
    stack_str_lst = []
    name = None
    description = None
    if topmost_frame is not None:
        try:
            try:
                frames_list = dbg.suspended_frames_manager.get_frames_list(thread_id)
                while frames_list is not None and len(frames_list):
                    frames = []
                    frame = None
                    if not name:
                        exc_type = frames_list.exc_type
                        if exc_type is not None:
                            try:
                                name = exc_type.__qualname__
                            except:
                                try:
                                    name = exc_type.__name__
                                except:
                                    try:
                                        name = str(exc_type)
                                    except:
                                        pass
                    if not description:
                        exc_desc = frames_list.exc_desc
                        if exc_desc is not None:
                            try:
                                description = str(exc_desc)
                            except:
                                pass
                    for frame_id, frame, method_name, original_filename, filename_in_utf8, lineno, _applied_mapping, show_as_current_frame, line_col_info in iter_visible_frames_info(dbg, frames_list):
                        line_text = linecache.getline(original_filename, lineno)
                        if not getattr(frame, 'IS_PLUGIN_FRAME', False):
                            if dbg.is_files_filter_enabled and dbg.apply_files_filter(frame, original_filename, False):
                                continue
                        if show_as_current_frame:
                            current_paused_frame_name = method_name
                            method_name += ' (Current frame)'
                        frames.append((filename_in_utf8, lineno, method_name, line_text, line_col_info))
                    if not source_path and frames:
                        source_path = frames[0][0]
                    if IS_PY311_OR_GREATER:
                        stack_summary = traceback.StackSummary()
                        for filename_in_utf8, lineno, method_name, line_text, line_col_info in frames[-max_frames:]:
                            frame_summary = traceback.FrameSummary(filename_in_utf8, lineno, method_name, line=line_text)
                            if line_col_info is not None:
                                frame_summary.end_lineno = line_col_info.end_lineno
                                frame_summary.colno = line_col_info.colno
                                frame_summary.end_colno = line_col_info.end_colno
                            stack_summary.append(frame_summary)
                        stack_str = ''.join(stack_summary.format())
                    else:
                        stack_str = ''.join(traceback.format_list((x[:-1] for x in frames[-max_frames:])))
                    try:
                        stype = frames_list.exc_type.__qualname__
                        smod = frames_list.exc_type.__module__
                        if smod not in ('__main__', 'builtins'):
                            if not isinstance(smod, str):
                                smod = '<unknown>'
                            stype = smod + '.' + stype
                    except Exception:
                        stype = '<unable to get exception type>'
                        pydev_log.exception('Error getting exception type.')
                    stack_str += '%s: %s\n' % (stype, frames_list.exc_desc)
                    stack_str += frames_list.exc_context_msg
                    stack_str_lst.append(stack_str)
                    frames_list = frames_list.chained_frames_list
                    if frames_list is None or not frames_list:
                        break
            except:
                pydev_log.exception('Error on build_exception_info_response.')
        finally:
            topmost_frame = None
    full_stack_str = ''.join(reversed(stack_str_lst))
    if not name:
        name = 'exception: type unknown'
    if not description:
        description = 'exception: no description'
    if current_paused_frame_name:
        name += '       (note: full exception trace is shown but execution is paused at: %s)' % (current_paused_frame_name,)
    if thread.stop_reason == CMD_STEP_CAUGHT_EXCEPTION:
        break_mode = pydevd_schema.ExceptionBreakMode.ALWAYS
    else:
        break_mode = pydevd_schema.ExceptionBreakMode.UNHANDLED
    response = pydevd_schema.ExceptionInfoResponse(request_seq=request_seq, success=True, command='exceptionInfo', body=pydevd_schema.ExceptionInfoResponseBody(exceptionId=name, description=description, breakMode=break_mode, details=pydevd_schema.ExceptionDetails(message=description, typeName=name, stackTrace=full_stack_str, source=source_path)))
    return response