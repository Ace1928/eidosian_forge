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
def ll_fz_clear_pixmap_with_value(pix, value):
    """
    Low-level wrapper for `::fz_clear_pixmap_with_value()`.
    Clears a pixmap with the given value.

    pix: The pixmap to clear.

    value: Values in the range 0 to 255 are valid. Each component
    sample for each pixel in the pixmap will be set to this value,
    while alpha will always be set to 255 (non-transparent).

    This function is horrible, and should be removed from the
    API and replaced with a less magic one.
    """
    return _mupdf.ll_fz_clear_pixmap_with_value(pix, value)