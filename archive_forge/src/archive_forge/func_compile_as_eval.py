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
def compile_as_eval(expression):
    """

    :param expression:
        The expression to be _compiled.

    :return: code object

    :raises Exception if the expression cannot be evaluated.
    """
    expression_to_evaluate = _expression_to_evaluate(expression)
    if _ASYNC_COMPILE_FLAGS is not None:
        return compile(expression_to_evaluate, '<string>', 'eval', _ASYNC_COMPILE_FLAGS)
    else:
        return compile(expression_to_evaluate, '<string>', 'eval')