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
def set_source_mapping(self, py_db, source_filename, mapping):
    """
        :param str source_filename:
            The filename for the source mapping (bytes on py2 and str on py3).
            This filename will be made absolute in this function.

        :param list(SourceMappingEntry) mapping:
            A list with the source mapping entries to be applied to the given filename.

        :return str:
            An error message if it was not possible to set the mapping or an empty string if
            everything is ok.
        """
    source_filename = self.filename_to_server(source_filename)
    absolute_source_filename = pydevd_file_utils.absolute_path(source_filename)
    for map_entry in mapping:
        map_entry.source_filename = absolute_source_filename
    error_msg = py_db.source_mapping.set_source_mapping(absolute_source_filename, mapping)
    if error_msg:
        return error_msg
    self.reapply_breakpoints(py_db)
    return ''