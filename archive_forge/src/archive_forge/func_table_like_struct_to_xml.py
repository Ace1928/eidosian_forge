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
def table_like_struct_to_xml(array, name, roffset, coffset, rows, cols, format):
    _, type_name, _ = get_type(array)
    if type_name == 'ndarray':
        array, metaxml, r, c, f = array_to_meta_xml(array, name, format)
        xml = metaxml
        format = '%' + f
        if rows == -1 and cols == -1:
            rows = r
            cols = c
        xml += array_to_xml(array, roffset, coffset, rows, cols, format)
    elif type_name == 'DataFrame':
        xml = dataframe_to_xml(array, name, roffset, coffset, rows, cols, format)
    else:
        raise VariableError('Do not know how to convert type %s to table' % type_name)
    return '<xml>%s</xml>' % xml