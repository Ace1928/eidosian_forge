from _pydevd_bundle.pydevd_constants import DebugInfoHolder, SHOW_COMPILE_CYTHON_COMMAND_LINE, NULL, LOG_TIME, \
from contextlib import contextmanager
import traceback
import os
import sys
import time
def show_compile_cython_command_line():
    if SHOW_COMPILE_CYTHON_COMMAND_LINE:
        dirname = os.path.dirname(os.path.dirname(__file__))
        error_once('warning: Debugger speedups using cython not found. Run \'"%s" "%s" build_ext --inplace\' to build.', sys.executable, os.path.join(dirname, 'setup_pydevd_cython.py'))