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
def ll_fz_install_load_system_font_funcs(f, f_cjk, f_fallback):
    """
    Low-level wrapper for `::fz_install_load_system_font_funcs()`.
    Install functions to allow MuPDF to request fonts from the
    system.

    Only one set of hooks can be in use at a time.
    """
    return _mupdf.ll_fz_install_load_system_font_funcs(f, f_cjk, f_fallback)