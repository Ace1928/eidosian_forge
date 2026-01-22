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
def pdf_to_string(self, sizep):
    """
        Class-aware wrapper for `::pdf_to_string()`.

        This method has out-params. Python/C# wrappers look like:
        	`pdf_to_string()` => `(const char *, size_t sizep)`
        """
    return _mupdf.PdfObj_pdf_to_string(self, sizep)