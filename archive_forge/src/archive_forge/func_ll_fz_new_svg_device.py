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
def ll_fz_new_svg_device(out, page_width, page_height, text_format, reuse_images):
    """
    Low-level wrapper for `::fz_new_svg_device()`.
    Create a device that outputs (single page) SVG files to
    the given output stream.

    Equivalent to fz_new_svg_device_with_id passing id = NULL.
    """
    return _mupdf.ll_fz_new_svg_device(out, page_width, page_height, text_format, reuse_images)