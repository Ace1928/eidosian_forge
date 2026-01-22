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
def internal_get_completions(dbg, seq, thread_id, frame_id, act_tok, line=-1, column=-1):
    """
    Note that if the column is >= 0, the act_tok is considered text and the actual
    activation token/qualifier is computed in this command.
    """
    try:
        remove_path = None
        try:
            qualifier = ''
            if column >= 0:
                token_and_qualifier = extract_token_and_qualifier(act_tok, line, column)
                act_tok = token_and_qualifier[0]
                if act_tok:
                    act_tok += '.'
                qualifier = token_and_qualifier[1]
            frame = dbg.find_frame(thread_id, frame_id)
            if frame is not None:
                completions = _pydev_completer.generate_completions(frame, act_tok)
                cmd = dbg.cmd_factory.make_get_completions_message(seq, completions, qualifier, start=column - len(qualifier))
                dbg.writer.add_command(cmd)
            else:
                cmd = dbg.cmd_factory.make_error_message(seq, 'internal_get_completions: Frame not found: %s from thread: %s' % (frame_id, thread_id))
                dbg.writer.add_command(cmd)
        finally:
            if remove_path is not None:
                sys.path.remove(remove_path)
    except:
        exc = get_exception_traceback_str()
        sys.stderr.write('%s\n' % (exc,))
        cmd = dbg.cmd_factory.make_error_message(seq, 'Error evaluating expression ' + exc)
        dbg.writer.add_command(cmd)