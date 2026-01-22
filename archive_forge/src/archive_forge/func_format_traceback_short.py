from types import TracebackType
from typing import List, Optional
import tempfile
import traceback
import contextlib
import inspect
import os.path
def format_traceback_short(tb):
    """Format a TracebackType in a short way, printing only the inner-most frame."""
    return format_frame(traceback.extract_tb(tb)[-1])