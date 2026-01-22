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
def pdf_lookup_metadata2(self, key):
    """
        Class-aware wrapper for `::pdf_lookup_metadata2()`.
        C++ alternative to `pdf_lookup_metadata()` that returns a `std::string`
        or calls `fz_throw()` if not found.
        """
    return _mupdf.PdfDocument_pdf_lookup_metadata2(self, key)