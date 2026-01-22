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
def fz_convert_color(self, sv, ds, dv, _is, params):
    """
        Class-aware wrapper for `::fz_convert_color()`.

        This method has out-params. Python/C# wrappers look like:
        	`fz_convert_color(const float *sv, ::fz_colorspace *ds, ::fz_colorspace *is, ::fz_color_params params)` => float dv

        	Convert color values sv from colorspace ss into colorvalues dv
        	for colorspace ds, via an optional intervening space is,
        	respecting the given color_params.
        """
    return _mupdf.FzColorspace_fz_convert_color(self, sv, ds, dv, _is, params)