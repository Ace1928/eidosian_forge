import pickle
from _pydevd_bundle.pydevd_constants import get_frame, get_current_thread_id, \
from _pydevd_bundle.pydevd_xml import ExceptionOnEvaluate, get_type, var_to_xml
from _pydev_bundle import pydev_log
import functools
from _pydevd_bundle.pydevd_thread_lifecycle import resume_threads, mark_thread_suspended, suspend_all_threads
from _pydevd_bundle.pydevd_comm_constants import CMD_SET_BREAK
import sys  # @Reimport
from _pydev_bundle._pydev_saved_modules import threading
from _pydevd_bundle import pydevd_save_locals, pydevd_timeout, pydevd_constants
from _pydev_bundle.pydev_imports import Exec, execfile
from _pydevd_bundle.pydevd_utils import to_string
import inspect
from _pydevd_bundle.pydevd_daemon_thread import PyDBDaemonThread
from _pydevd_bundle.pydevd_save_locals import update_globals_and_locals
from functools import lru_cache
def custom_operation(dbg, thread_id, frame_id, scope, attrs, style, code_or_file, operation_fn_name):
    """
    We'll execute the code_or_file and then search in the namespace the operation_fn_name to execute with the given var.

    code_or_file: either some code (i.e.: from pprint import pprint) or a file to be executed.
    operation_fn_name: the name of the operation to execute after the exec (i.e.: pprint)
    """
    expressionValue = getVariable(dbg, thread_id, frame_id, scope, attrs)
    try:
        namespace = {'__name__': '<custom_operation>'}
        if style == 'EXECFILE':
            namespace['__file__'] = code_or_file
            execfile(code_or_file, namespace, namespace)
        else:
            namespace['__file__'] = '<customOperationCode>'
            Exec(code_or_file, namespace, namespace)
        return str(namespace[operation_fn_name](expressionValue))
    except:
        pydev_log.exception()