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
def request_set_next(self, py_db, seq, thread_id, set_next_cmd_id, original_filename, line, func_name):
    """
        set_next_cmd_id may actually be one of:

        CMD_RUN_TO_LINE
        CMD_SET_NEXT_STATEMENT

        CMD_SMART_STEP_INTO -- note: request_smart_step_into is preferred if it's possible
                               to work with bytecode offset.

        :param Optional[str] original_filename:
            If available, the filename may be source translated, otherwise no translation will take
            place (the set next just needs the line afterwards as it executes locally, but for
            the Jupyter integration, the source mapping may change the actual lines and not only
            the filename).
        """
    t = pydevd_find_thread_by_id(thread_id)
    if t:
        if original_filename is not None:
            translated_filename = self.filename_to_server(original_filename)
            pydev_log.debug('Set next (after path translation) in: %s line: %s', translated_filename, line)
            func_name = self.to_str(func_name)
            assert translated_filename.__class__ == str
            assert func_name.__class__ == str
            _source_mapped_filename, new_line, multi_mapping_applied = py_db.source_mapping.map_to_server(translated_filename, line)
            if multi_mapping_applied:
                pydev_log.debug('Set next (after source mapping) in: %s line: %s', translated_filename, line)
                line = new_line
        int_cmd = InternalSetNextStatementThread(thread_id, set_next_cmd_id, line, func_name, seq=seq)
        py_db.post_internal_command(int_cmd, thread_id)
    elif thread_id.startswith('__frame__:'):
        sys.stderr.write("Can't set next statement in tasklet: %s\n" % (thread_id,))