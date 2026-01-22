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
def ll_fz_format_link_uri(doc, dest):
    """
    Low-level wrapper for `::fz_format_link_uri()`.
    Format an internal link to a page number, location, and possible viewing parameters,
    suitable for use with fz_create_link.

    Returns a newly allocated string that the caller must free.
    """
    return _mupdf.ll_fz_format_link_uri(doc, dest)