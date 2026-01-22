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
def ll_fz_lookup_metadata2(doc, key):
    """
     Low-level wrapper for `::fz_lookup_metadata2()`.
    C++ alternative to `fz_lookup_metadata()` that returns a `std::string`
    or calls `fz_throw()` if not found.
    """
    return _mupdf.ll_fz_lookup_metadata2(doc, key)