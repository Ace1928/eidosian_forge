import os
import re
import sys
from _pydev_bundle._pydev_saved_modules import threading
from _pydevd_bundle.pydevd_constants import get_global_debugger, IS_WINDOWS, IS_JYTHON, get_current_thread_id, \
from _pydev_bundle import pydev_log
from contextlib import contextmanager
from _pydevd_bundle import pydevd_constants, pydevd_defaults
from _pydevd_bundle.pydevd_defaults import PydevdCustomization
import ast
def create_warn_fork_exec(original_name):
    """
    _posixsubprocess.fork_exec(args, executable_list, close_fds, ... (13 more))
    """

    def new_warn_fork_exec(*args):
        try:
            import _posixsubprocess
            warn_multiproc()
            return getattr(_posixsubprocess, original_name)(*args)
        except:
            pass
    return new_warn_fork_exec