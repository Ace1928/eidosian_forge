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
def fz_separation_equivalent(self, idx, dst_cs, dst_color, prf, color_params):
    """
        Class-aware wrapper for `::fz_separation_equivalent()`.

        This method has out-params. Python/C# wrappers look like:
        	`fz_separation_equivalent(int idx, ::fz_colorspace *dst_cs, ::fz_colorspace *prf, ::fz_color_params color_params)` => float dst_color

        	Get the equivalent separation color in a given colorspace.
        """
    return _mupdf.FzSeparations_fz_separation_equivalent(self, idx, dst_cs, dst_color, prf, color_params)