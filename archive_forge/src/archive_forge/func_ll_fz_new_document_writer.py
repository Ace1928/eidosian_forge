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
def ll_fz_new_document_writer(path, format, options):
    """
    Low-level wrapper for `::fz_new_document_writer()`.
    Create a new fz_document_writer, for a
    file of the given type.

    path: The document name to write (or NULL for default)

    format: Which format to write (currently cbz, html, pdf, pam,
    pbm, pgm, pkm, png, ppm, pnm, svg, text, xhtml, docx, odt)

    options: NULL, or pointer to comma separated string to control
    file generation.
    """
    return _mupdf.ll_fz_new_document_writer(path, format, options)