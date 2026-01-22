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
def fz_new_draw_device_with_options(options, mediabox, pixmap):
    """
    Class-aware wrapper for `::fz_new_draw_device_with_options()`.

    This function has out-params. Python/C# wrappers look like:
    	`fz_new_draw_device_with_options(const ::fz_draw_options *options, ::fz_rect mediabox, ::fz_pixmap **pixmap)` => `(fz_device *)`

    	Create a new pixmap and draw device, using the specified options.

    	options: Options to configure the draw device, and choose the
    	resolution and colorspace.

    	mediabox: The bounds of the page in points.

    	pixmap: An out parameter containing the newly created pixmap.
    """
    return _mupdf.fz_new_draw_device_with_options(options, mediabox, pixmap)