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
def ll_fz_load_system_cjk_font(name, ordering, serif):
    """
    Low-level wrapper for `::fz_load_system_cjk_font()`.
    Attempt to load a given font from
    the system.

    name: The name of the desired font.

    ordering: The ordering to load the font from (e.g. FZ_ADOBE_KOREA)

    serif: 1 if serif desired, 0 otherwise.

    Returns a new font handle, or NULL if no matching font was found
    (or on error).
    """
    return _mupdf.ll_fz_load_system_cjk_font(name, ordering, serif)