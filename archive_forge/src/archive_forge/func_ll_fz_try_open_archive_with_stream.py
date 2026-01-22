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
def ll_fz_try_open_archive_with_stream(file):
    """
    Low-level wrapper for `::fz_try_open_archive_with_stream()`.
    Open zip or tar archive stream.

    Does the same as fz_open_archive_with_stream, but will not throw
    an error in the event of failing to recognise the format. Will
    still throw errors in other cases though!
    """
    return _mupdf.ll_fz_try_open_archive_with_stream(file)