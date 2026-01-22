import sys
import bisect
import types
from _pydev_bundle._pydev_saved_modules import threading
from _pydevd_bundle import pydevd_utils, pydevd_source_mapping
from _pydevd_bundle.pydevd_additional_thread_info import set_additional_thread_info
from _pydevd_bundle.pydevd_comm import (InternalGetThreadStack, internal_get_completions,
from _pydevd_bundle.pydevd_comm_constants import (CMD_THREAD_SUSPEND, file_system_encoding,
from _pydevd_bundle.pydevd_constants import (get_current_thread_id, set_protocol, get_protocol,
from _pydevd_bundle.pydevd_net_command_factory_json import NetCommandFactoryJson
from _pydevd_bundle.pydevd_net_command_factory_xml import NetCommandFactory
import pydevd_file_utils
from _pydev_bundle import pydev_log
from _pydevd_bundle.pydevd_breakpoints import LineBreakpoint
from pydevd_tracing import get_exception_traceback_str
import os
import subprocess
import ctypes
from _pydevd_bundle.pydevd_collect_bytecode_info import code_to_bytecode_representation
import itertools
import linecache
from _pydevd_bundle.pydevd_utils import DAPGrouper, interrupt_main_thread
from _pydevd_bundle.pydevd_daemon_thread import run_as_pydevd_daemon_thread
from _pydevd_bundle.pydevd_thread_lifecycle import pydevd_find_thread_by_id, resume_threads
import tokenize
def remove_breakpoint(self, py_db, received_filename, breakpoint_type, breakpoint_id):
    """
        :param str received_filename:
            Note: must be sent as it was received in the protocol. It may be translated in this
            function.

        :param str breakpoint_type:
            One of: 'python-line', 'django-line', 'jinja2-line'.

        :param int breakpoint_id:
        """
    received_filename_normalized = pydevd_file_utils.normcase_from_client(received_filename)
    for key, val in list(py_db.api_received_breakpoints.items()):
        original_filename_normalized, existing_breakpoint_id = key
        _new_filename, _api_add_breakpoint_params = val
        if received_filename_normalized == original_filename_normalized and existing_breakpoint_id == breakpoint_id:
            del py_db.api_received_breakpoints[key]
            break
    else:
        pydev_log.info('Did not find breakpoint to remove: %s (breakpoint id: %s)', received_filename, breakpoint_id)
    file_to_id_to_breakpoint = None
    received_filename = self.filename_to_server(received_filename)
    canonical_normalized_filename = pydevd_file_utils.canonical_normalized_path(received_filename)
    if breakpoint_type == 'python-line':
        file_to_line_to_breakpoints = py_db.breakpoints
        file_to_id_to_breakpoint = py_db.file_to_id_to_line_breakpoint
    elif py_db.plugin is not None:
        result = py_db.plugin.get_breakpoints(py_db, breakpoint_type)
        if result is not None:
            file_to_id_to_breakpoint = py_db.file_to_id_to_plugin_breakpoint
            file_to_line_to_breakpoints = result
    if file_to_id_to_breakpoint is None:
        pydev_log.critical('Error removing breakpoint. Cannot handle breakpoint of type %s', breakpoint_type)
    else:
        try:
            id_to_pybreakpoint = file_to_id_to_breakpoint.get(canonical_normalized_filename, {})
            if DebugInfoHolder.DEBUG_TRACE_LEVEL >= 1:
                existing = id_to_pybreakpoint[breakpoint_id]
                pydev_log.info('Removed breakpoint:%s - line:%s - func_name:%s (id: %s)\n' % (canonical_normalized_filename, existing.line, existing.func_name, breakpoint_id))
            del id_to_pybreakpoint[breakpoint_id]
            py_db.consolidate_breakpoints(canonical_normalized_filename, id_to_pybreakpoint, file_to_line_to_breakpoints)
            if py_db.plugin is not None:
                py_db.has_plugin_line_breaks = py_db.plugin.has_line_breaks()
                py_db.plugin.after_breakpoints_consolidated(py_db, canonical_normalized_filename, id_to_pybreakpoint, file_to_line_to_breakpoints)
        except KeyError:
            pydev_log.info('Error removing breakpoint: Breakpoint id not found: %s id: %s. Available ids: %s\n', canonical_normalized_filename, breakpoint_id, list(id_to_pybreakpoint))
    py_db.on_breakpoints_changed(removed=True)