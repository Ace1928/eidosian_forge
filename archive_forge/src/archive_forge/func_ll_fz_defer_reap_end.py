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
def ll_fz_defer_reap_end():
    """
    Low-level wrapper for `::fz_defer_reap_end()`.
    Decrement the defer reap count.

    If the defer reap count returns to 0, and the store
    has reapable objects in, a reap pass will begin.

    Call this at the end of a process during which you
    potentially might drop many reapable objects.

    It is vital that every fz_defer_reap_start is matched
    by a fz_defer_reap_end call.
    """
    return _mupdf.ll_fz_defer_reap_end()