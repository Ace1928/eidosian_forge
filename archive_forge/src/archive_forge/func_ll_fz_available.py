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
def ll_fz_available(stm, max):
    """
    Low-level wrapper for `::fz_available()`.
    Ask how many bytes are available immediately from
    a given stream.

    stm: The stream to read from.

    max: A hint for the underlying stream; the maximum number of
    bytes that we are sure we will want to read. If you do not know
    this number, give 1.

    Returns the number of bytes immediately available between the
    read and write pointers. This number is guaranteed only to be 0
    if we have hit EOF. The number of bytes returned here need have
    no relation to max (could be larger, could be smaller).
    """
    return _mupdf.ll_fz_available(stm, max)