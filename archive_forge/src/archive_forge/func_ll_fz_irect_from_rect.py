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
def ll_fz_irect_from_rect(rect):
    """
    Low-level wrapper for `::fz_irect_from_rect()`.
    Convert a rect into the minimal bounding box
    that covers the rectangle.

    Coordinates in a bounding box are integers, so rounding of the
    rects coordinates takes place. The top left corner is rounded
    upwards and left while the bottom right corner is rounded
    downwards and to the right.
    """
    return _mupdf.ll_fz_irect_from_rect(rect)