from _pydevd_bundle.pydevd_constants import DebugInfoHolder, SHOW_COMPILE_CYTHON_COMMAND_LINE, NULL, LOG_TIME, \
from contextlib import contextmanager
import traceback
import os
import sys
import time
def debug_once(msg, *args):
    if DebugInfoHolder.DEBUG_TRACE_LEVEL >= 3:
        error_once(msg, *args)