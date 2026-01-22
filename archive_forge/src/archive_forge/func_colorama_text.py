import atexit
import contextlib
import sys
from .ansitowin32 import AnsiToWin32
@contextlib.contextmanager
def colorama_text(*args, **kwargs):
    init(*args, **kwargs)
    try:
        yield
    finally:
        deinit()