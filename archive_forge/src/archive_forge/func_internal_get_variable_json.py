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
@silence_warnings_decorator
def internal_get_variable_json(py_db, request):
    """
        :param VariablesRequest request:
    """
    arguments = request.arguments
    variables_reference = arguments.variablesReference
    scope = None
    if isinstance_checked(variables_reference, ScopeRequest):
        scope = variables_reference
        variables_reference = variables_reference.variable_reference
    fmt = arguments.format
    if hasattr(fmt, 'to_dict'):
        fmt = fmt.to_dict()
    variables = []
    try:
        try:
            variable = py_db.suspended_frames_manager.get_variable(variables_reference)
        except KeyError:
            pass
        else:
            for child_var in variable.get_children_variables(fmt=fmt, scope=scope):
                variables.append(child_var.get_var_data(fmt=fmt))
    except:
        try:
            exc, exc_type, tb = sys.exc_info()
            err = ''.join(traceback.format_exception(exc, exc_type, tb))
            variables = [{'name': '<error>', 'value': err, 'type': '<error>', 'variablesReference': 0}]
        except:
            err = '<Internal error - unable to get traceback when getting variables>'
            pydev_log.exception(err)
            variables = []
    body = VariablesResponseBody(variables)
    variables_response = pydevd_base_schema.build_response(request, kwargs={'body': body})
    py_db.writer.add_command(NetCommand(CMD_RETURN, 0, variables_response, is_json=True))