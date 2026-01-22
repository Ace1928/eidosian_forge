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
def ll_fz_open_zip_archive(path):
    """
    Low-level wrapper for `::fz_open_zip_archive()`.
    Open a zip archive file.

    An exception is thrown if the file is not a zip archive as
    indicated by the presence of a zip signature.

    filename: a path to a zip archive file as it would be given to
    open(2).
    """
    return _mupdf.ll_fz_open_zip_archive(path)