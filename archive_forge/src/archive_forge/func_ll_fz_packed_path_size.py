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
def ll_fz_packed_path_size(path):
    """
    Low-level wrapper for `::fz_packed_path_size()`.
    Return the number of bytes required to pack a path.
    """
    return _mupdf.ll_fz_packed_path_size(path)