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
def ll_fz_drop_zip_writer(zip):
    """
    Low-level wrapper for `::fz_drop_zip_writer()`.
    Drop the reference to the zipfile.

    In common with other 'drop' methods, this will never throw an
    exception.
    """
    return _mupdf.ll_fz_drop_zip_writer(zip)