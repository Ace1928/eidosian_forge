from sys import version_info as _swig_python_version_info
import weakref
import inspect
import os
import re
import sys
import traceback
import inspect
import io
import os
import sys
import traceback
import types
def ll_fz_new_stream(state, next, drop):
    """
    Low-level wrapper for `::fz_new_stream()`.
    Create a new stream object with the given
    internal state and function pointers.

    state: Internal state (opaque to everything but implementation).

    next: Should provide the next set of bytes (up to max) of stream
    data. Return the number of bytes read, or EOF when there is no
    more data.

    drop: Should clean up and free the internal state. May not
    throw exceptions.
    """
    return _mupdf.ll_fz_new_stream(state, next, drop)