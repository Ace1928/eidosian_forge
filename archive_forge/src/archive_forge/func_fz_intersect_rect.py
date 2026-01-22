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
def fz_intersect_rect(self, *args):
    """
        *Overload 1:*
         Class-aware wrapper for `::fz_intersect_rect()`.
        		Compute intersection of two rectangles.

        		Given two rectangles, update the first to be the smallest
        		axis-aligned rectangle that covers the area covered by both
        		given rectangles. If either rectangle is empty then the
        		intersection is also empty. If either rectangle is infinite
        		then the intersection is simply the non-infinite rectangle.
        		Should both rectangles be infinite, then the intersection is
        		also infinite.


        |

        *Overload 2:*
         Class-aware wrapper for `::fz_intersect_rect()`.
        		Compute intersection of two rectangles.

        		Given two rectangles, update the first to be the smallest
        		axis-aligned rectangle that covers the area covered by both
        		given rectangles. If either rectangle is empty then the
        		intersection is also empty. If either rectangle is infinite
        		then the intersection is simply the non-infinite rectangle.
        		Should both rectangles be infinite, then the intersection is
        		also infinite.
        """
    return _mupdf.FzRect_fz_intersect_rect(self, *args)