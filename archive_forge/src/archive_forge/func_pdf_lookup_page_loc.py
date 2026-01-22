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
def pdf_lookup_page_loc(self, needle, parentp, indexp):
    """
        Class-aware wrapper for `::pdf_lookup_page_loc()`.

        This method has out-params. Python/C# wrappers look like:
        	`pdf_lookup_page_loc(int needle, ::pdf_obj **parentp)` => `(pdf_obj *, int indexp)`
        """
    return _mupdf.PdfDocument_pdf_lookup_page_loc(self, needle, parentp, indexp)