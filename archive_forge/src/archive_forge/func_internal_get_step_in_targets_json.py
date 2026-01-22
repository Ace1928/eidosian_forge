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
def internal_get_step_in_targets_json(dbg, seq, thread_id, frame_id, request, set_additional_thread_info):
    try:
        thread = pydevd_find_thread_by_id(thread_id)
        frame = dbg.find_frame(thread_id, frame_id)
        if thread is None or frame is None:
            body = StepInTargetsResponseBody([])
            variables_response = pydevd_base_schema.build_response(request, kwargs={'body': body, 'success': False, 'message': 'Thread to get step in targets seems to have resumed already.'})
            cmd = NetCommand(CMD_RETURN, 0, variables_response, is_json=True)
            dbg.writer.add_command(cmd)
            return
        start_line = 0
        end_line = 99999999
        if pydevd_bytecode_utils is None:
            variants = []
        else:
            variants = pydevd_bytecode_utils.calculate_smart_step_into_variants(frame, start_line, end_line)
        info = set_additional_thread_info(thread)
        targets = []
        counter = itertools.count(0)
        target_id_to_variant = {}
        for variant in variants:
            if not variant.is_visited:
                if variant.children_variants:
                    for child_variant in variant.children_variants:
                        target_id = next(counter)
                        if child_variant.call_order > 1:
                            targets.append(StepInTarget(id=target_id, label='%s (call %s)' % (child_variant.name, child_variant.call_order)))
                        else:
                            targets.append(StepInTarget(id=target_id, label=child_variant.name))
                        target_id_to_variant[target_id] = child_variant
                        if len(targets) >= 15:
                            break
                else:
                    target_id = next(counter)
                    if variant.call_order > 1:
                        targets.append(StepInTarget(id=target_id, label='%s (call %s)' % (variant.name, variant.call_order)))
                    else:
                        targets.append(StepInTarget(id=target_id, label=variant.name))
                    target_id_to_variant[target_id] = variant
                    if len(targets) >= 15:
                        break
        info.pydev_smart_step_into_variants = tuple(variants)
        info.target_id_to_smart_step_into_variant = target_id_to_variant
        body = StepInTargetsResponseBody(targets=targets)
        response = pydevd_base_schema.build_response(request, kwargs={'body': body})
        cmd = NetCommand(CMD_RETURN, 0, response, is_json=True)
        dbg.writer.add_command(cmd)
    except Exception as e:
        pydev_log.exception('Error calculating Smart Step Into Variants.')
        body = StepInTargetsResponseBody([])
        variables_response = pydevd_base_schema.build_response(request, kwargs={'body': body, 'success': False, 'message': str(e)})
        cmd = NetCommand(CMD_RETURN, 0, variables_response, is_json=True)
        dbg.writer.add_command(cmd)