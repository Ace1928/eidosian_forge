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
def pdf_eval_function(self, _in, inlen, out, outlen):
    """
        Class-aware wrapper for `::pdf_eval_function()`.

        This method has out-params. Python/C# wrappers look like:
        	`pdf_eval_function(const float *in, int inlen, int outlen)` => float out
        """
    return _mupdf.PdfFunction_pdf_eval_function(self, _in, inlen, out, outlen)