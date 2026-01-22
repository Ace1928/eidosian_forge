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
def create_CreateProcess(original_name):
    """
    CreateProcess(*args, **kwargs)
    """

    def new_CreateProcess(app_name, cmd_line, *args):
        try:
            import _subprocess
        except ImportError:
            import _winapi as _subprocess
        if _get_apply_arg_patching():
            cmd_line = patch_arg_str_win(cmd_line)
            send_process_created_message()
        return getattr(_subprocess, original_name)(app_name, cmd_line, *args)
    return new_CreateProcess