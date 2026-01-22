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
def ll_fz_flush_warnings():
    """
    Low-level wrapper for `::fz_flush_warnings()`.
    Flush any repeated warnings.

    Repeated warnings are buffered, counted and eventually printed
    along with the number of repetitions. Call fz_flush_warnings
    to force printing of the latest buffered warning and the
    number of repetitions, for example to make sure that all
    warnings are printed before exiting an application.
    """
    return _mupdf.ll_fz_flush_warnings()