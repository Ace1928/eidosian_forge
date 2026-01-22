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
def fz_round_rect(self):
    """
        Class-aware wrapper for `::fz_round_rect()`.
        	Round rectangle coordinates.

        	Coordinates in a bounding box are integers, so rounding of the
        	rects coordinates takes place. The top left corner is rounded
        	upwards and left while the bottom right corner is rounded
        	downwards and to the right.

        	This differs from fz_irect_from_rect, in that fz_irect_from_rect
        	slavishly follows the numbers (i.e any slight over/under
        	calculations can cause whole extra pixels to be added).
        	fz_round_rect allows for a small amount of rounding error when
        	calculating the bbox.
        """
    return _mupdf.FzRect_fz_round_rect(self)