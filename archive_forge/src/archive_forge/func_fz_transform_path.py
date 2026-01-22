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
def fz_transform_path(self, transform):
    """
        Class-aware wrapper for `::fz_transform_path()`.
        	Transform a path by a given
        	matrix.

        	path: The path to modify (must not be a packed path).

        	transform: The transform to apply.

        	Throws exceptions if the path is packed, or on failure
        	to allocate.
        """
    return _mupdf.FzPath_fz_transform_path(self, transform)