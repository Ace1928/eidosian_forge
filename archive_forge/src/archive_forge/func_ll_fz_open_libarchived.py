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
def ll_fz_open_libarchived(chain):
    """
    Low-level wrapper for `::fz_open_libarchived()`.
    libarchived filter performs generic compressed decoding of data
    in any format understood by libarchive from the chained filter.

    This will throw an exception if libarchive is not built in, or
    if the compression format is not recognised.
    """
    return _mupdf.ll_fz_open_libarchived(chain)