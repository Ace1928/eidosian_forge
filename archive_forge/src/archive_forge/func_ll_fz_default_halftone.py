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
def ll_fz_default_halftone(num_comps):
    """
    Low-level wrapper for `::fz_default_halftone()`.
    Create a 'default' halftone structure
    for the given number of components.

    num_comps: The number of components to use.

    Returns a simple default halftone. The default halftone uses
    the same halftone tile for each plane, which may not be ideal
    for all purposes.
    """
    return _mupdf.ll_fz_default_halftone(num_comps)