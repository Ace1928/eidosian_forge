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
def change_attr_expression(frame, attr, expression, dbg, value=SENTINEL_VALUE):
    """Changes some attribute in a given frame.
    """
    if frame is None:
        return
    try:
        expression = expression.replace('@LINE@', '\n')
        if dbg.plugin and value is SENTINEL_VALUE:
            result = dbg.plugin.change_variable(frame, attr, expression)
            if result:
                return result
        if attr[:7] == 'Globals':
            attr = attr[8:]
            if attr in frame.f_globals:
                if value is SENTINEL_VALUE:
                    value = eval(expression, frame.f_globals, frame.f_locals)
                frame.f_globals[attr] = value
                return frame.f_globals[attr]
        else:
            if '.' not in attr:
                if pydevd_save_locals.is_save_locals_available():
                    if value is SENTINEL_VALUE:
                        value = eval(expression, frame.f_globals, frame.f_locals)
                    frame.f_locals[attr] = value
                    pydevd_save_locals.save_locals(frame)
                    return frame.f_locals[attr]
            if value is SENTINEL_VALUE:
                value = eval(expression, frame.f_globals, frame.f_locals)
            result = value
            Exec('%s=%s' % (attr, expression), frame.f_globals, frame.f_locals)
            return result
    except Exception:
        pydev_log.exception()