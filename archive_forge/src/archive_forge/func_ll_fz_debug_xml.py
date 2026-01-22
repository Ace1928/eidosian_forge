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
def ll_fz_debug_xml(item, level):
    """
    Low-level wrapper for `::fz_debug_xml()`.
    Pretty-print an XML tree to stdout. (Deprecated, use
    fz_output_xml in preference).
    """
    return _mupdf.ll_fz_debug_xml(item, level)