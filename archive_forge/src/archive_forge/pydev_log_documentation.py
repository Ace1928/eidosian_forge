from _pydevd_bundle.pydevd_constants import DebugInfoHolder, SHOW_COMPILE_CYTHON_COMMAND_LINE, NULL, LOG_TIME, \
from contextlib import contextmanager
import traceback
import os
import sys
import time

    Levels are:

    0 most serious warnings/errors (always printed)
    1 warnings/significant events
    2 informational trace
    3 verbose mode
    