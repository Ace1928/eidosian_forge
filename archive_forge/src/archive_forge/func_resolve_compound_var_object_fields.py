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
def resolve_compound_var_object_fields(var, attrs):
    """
    Resolve compound variable by its object and attributes

    :param var: an object of variable
    :param attrs: a sequence of variable's attributes separated by 	 (i.e.: obj	attr1	attr2)
    :return: a dictionary of variables's fields
    """
    attr_list = attrs.split('\t')
    for k in attr_list:
        type, _type_name, resolver = get_type(var)
        var = resolver.resolve(var, k)
    try:
        type, _type_name, resolver = get_type(var)
        return resolver.get_dictionary(var)
    except:
        pydev_log.exception()