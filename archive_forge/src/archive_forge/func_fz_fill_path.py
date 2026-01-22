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
def fz_fill_path(self, path, even_odd, ctm, colorspace, color, alpha, color_params):
    """
        Class-aware wrapper for `::fz_fill_path()`.
        	Device calls; graphics primitives and containers.
        """
    return _mupdf.FzDevice_fz_fill_path(self, path, even_odd, ctm, colorspace, color, alpha, color_params)