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
def pdf_annot_line_ending_styles(self, start_style, end_style):
    """
        Class-aware wrapper for `::pdf_annot_line_ending_styles()`.

        This method has out-params. Python/C# wrappers look like:
        	`pdf_annot_line_ending_styles()` => `(enum pdf_line_ending start_style, enum pdf_line_ending end_style)`
        """
    return _mupdf.PdfAnnot_pdf_annot_line_ending_styles(self, start_style, end_style)