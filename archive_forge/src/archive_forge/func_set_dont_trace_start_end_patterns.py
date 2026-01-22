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
def set_dont_trace_start_end_patterns(self, py_db, start_patterns, end_patterns):
    start_patterns = tuple((pydevd_file_utils.normcase(x) for x in start_patterns))
    end_patterns = tuple((pydevd_file_utils.normcase(x) for x in end_patterns))
    reset_caches = False
    dont_trace_start_end_patterns_previously_set = py_db.dont_trace_external_files.__name__ == 'custom_dont_trace_external_files'
    if not dont_trace_start_end_patterns_previously_set and (not start_patterns) and (not end_patterns):
        return
    if not py_db.is_cache_file_type_empty():
        if dont_trace_start_end_patterns_previously_set:
            reset_caches = py_db.dont_trace_external_files.start_patterns != start_patterns or py_db.dont_trace_external_files.end_patterns != end_patterns
        else:
            reset_caches = True

    def custom_dont_trace_external_files(abs_path):
        normalized_abs_path = pydevd_file_utils.normcase(abs_path)
        return normalized_abs_path.startswith(start_patterns) or normalized_abs_path.endswith(end_patterns)
    custom_dont_trace_external_files.start_patterns = start_patterns
    custom_dont_trace_external_files.end_patterns = end_patterns
    py_db.dont_trace_external_files = custom_dont_trace_external_files
    if reset_caches:
        py_db.clear_dont_trace_start_end_patterns_caches()