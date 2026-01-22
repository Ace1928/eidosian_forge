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
def fz_load_system_font(name, bold, italic, needs_exact_metrics):
    """
    Class-aware wrapper for `::fz_load_system_font()`.
    	Attempt to load a given font from the system.

    	name: The name of the desired font.

    	bold: 1 if bold desired, 0 otherwise.

    	italic: 1 if italic desired, 0 otherwise.

    	needs_exact_metrics: 1 if an exact metrical match is required,
    	0 otherwise.

    	Returns a new font handle, or NULL if no matching font was found
    	(or on error).
    """
    return _mupdf.fz_load_system_font(name, bold, italic, needs_exact_metrics)