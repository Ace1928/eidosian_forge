import fnmatch
import sys
import os
from inspect import CO_GENERATOR, CO_COROUTINE, CO_ASYNC_GENERATOR
def set_until(self, frame, lineno=None):
    """Stop when the line with the lineno greater than the current one is
        reached or when returning from current frame."""
    if lineno is None:
        lineno = frame.f_lineno + 1
    self._set_stopinfo(frame, frame, lineno)