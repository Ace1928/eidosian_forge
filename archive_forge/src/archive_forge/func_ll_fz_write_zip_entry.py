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
def ll_fz_write_zip_entry(zip, name, buf, compress):
    """
    Low-level wrapper for `::fz_write_zip_entry()`.
    Given a buffer of data, (optionally) compress it, and add it to
    the zip file with the given name.
    """
    return _mupdf.ll_fz_write_zip_entry(zip, name, buf, compress)