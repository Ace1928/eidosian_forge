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
def pdf_decode_cmap(self, s, e, cpt):
    """
        Class-aware wrapper for `::pdf_decode_cmap()`.

        This method has out-params. Python/C# wrappers look like:
        	`pdf_decode_cmap(unsigned char *s, unsigned char *e)` => `(int, unsigned int cpt)`
        """
    return _mupdf.PdfCmap_pdf_decode_cmap(self, s, e, cpt)