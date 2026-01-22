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
def ll_fz_new_pdfocr_band_writer(out, options):
    """
    Low-level wrapper for `::fz_new_pdfocr_band_writer()`.
    Create a new band writer, outputing pdfocr.

    Ownership of output stays with the caller, the band writer
    borrows the reference. The caller must keep the output around
    for the duration of the band writer, and then close/drop as
    appropriate.
    """
    return _mupdf.ll_fz_new_pdfocr_band_writer(out, options)