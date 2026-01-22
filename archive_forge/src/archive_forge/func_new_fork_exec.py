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
def new_fork_exec(args, *other_args):
    import subprocess
    if _get_apply_arg_patching():
        args = patch_args(args)
        send_process_created_message()
    return getattr(subprocess, original_name)(args, *other_args)