from _pydevd_bundle.pydevd_constants import DebugInfoHolder, SHOW_COMPILE_CYTHON_COMMAND_LINE, NULL, LOG_TIME, \
from contextlib import contextmanager
import traceback
import os
import sys
import time
@contextmanager
def log_context(trace_level, stream):
    """
    To be used to temporarily change the logging settings.
    """
    with _LoggingGlobals._initialize_lock:
        original_trace_level = DebugInfoHolder.DEBUG_TRACE_LEVEL
        original_debug_stream = _LoggingGlobals._debug_stream
        original_pydevd_debug_file = DebugInfoHolder.PYDEVD_DEBUG_FILE
        original_debug_stream_filename = _LoggingGlobals._debug_stream_filename
        original_initialized = _LoggingGlobals._debug_stream_initialized
        DebugInfoHolder.DEBUG_TRACE_LEVEL = trace_level
        _LoggingGlobals._debug_stream = stream
        _LoggingGlobals._debug_stream_initialized = True
    try:
        yield
    finally:
        with _LoggingGlobals._initialize_lock:
            DebugInfoHolder.DEBUG_TRACE_LEVEL = original_trace_level
            _LoggingGlobals._debug_stream = original_debug_stream
            DebugInfoHolder.PYDEVD_DEBUG_FILE = original_pydevd_debug_file
            _LoggingGlobals._debug_stream_filename = original_debug_stream_filename
            _LoggingGlobals._debug_stream_initialized = original_initialized