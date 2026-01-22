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
def fz_adjust_rect_for_stroke(self, stroke, ctm):
    """
        Class-aware wrapper for `::fz_adjust_rect_for_stroke()`.
        	Given a rectangle (assumed to be the bounding box for a path),
        	expand it to allow for the expansion of the bbox that would be
        	seen by stroking the path with the given stroke state and
        	transform.
        """
    return _mupdf.FzRect_fz_adjust_rect_for_stroke(self, stroke, ctm)