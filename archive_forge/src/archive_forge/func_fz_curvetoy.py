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
def fz_curvetoy(self, x0, y0, x2, y2):
    """
        Class-aware wrapper for `::fz_curvetoy()`.
        	Append a 'curvetoy' command to an open path. (For a
        	cubic bezier with the second control coordinate equal to
        	the end point).

        	path: The path to modify.

        	x0, y0: The coordinates of the first control point for the
        	curve.

        	x2, y2: The end coordinates for the curve (and the second
        	control coordinate).

        	Throws exceptions on failure to allocate, or attempting to
        	modify a packed path.
        """
    return _mupdf.FzPath_fz_curvetoy(self, x0, y0, x2, y2)