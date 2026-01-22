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
def pdf_page_presentation(self, transition, duration):
    """
        Class-aware wrapper for `::pdf_page_presentation()`.

        This method has out-params. Python/C# wrappers look like:
        	`pdf_page_presentation(::fz_transition *transition)` => `(fz_transition *, float duration)`
        """
    return _mupdf.PdfPage_pdf_page_presentation(self, transition, duration)