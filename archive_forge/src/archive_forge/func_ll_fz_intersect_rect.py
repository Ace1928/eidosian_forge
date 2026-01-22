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
def ll_fz_intersect_rect(a, b):
    """
    Low-level wrapper for `::fz_intersect_rect()`.
    Compute intersection of two rectangles.

    Given two rectangles, update the first to be the smallest
    axis-aligned rectangle that covers the area covered by both
    given rectangles. If either rectangle is empty then the
    intersection is also empty. If either rectangle is infinite
    then the intersection is simply the non-infinite rectangle.
    Should both rectangles be infinite, then the intersection is
    also infinite.
    """
    return _mupdf.ll_fz_intersect_rect(a, b)