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
def ll_fz_search_page2(doc, number, needle, hit_max):
    """
     Low-level wrapper for `::fz_search_page2()`.
    C++ alternative to fz_search_page() that returns information in a std::vector.
    """
    return _mupdf.ll_fz_search_page2(doc, number, needle, hit_max)