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
def fz_new_bbox_device(self):
    """
        Class-aware wrapper for `::fz_new_bbox_device()`.
        	Create a device to compute the bounding
        	box of all marks on a page.

        	The returned bounding box will be the union of all bounding
        	boxes of all objects on a page.
        """
    return _mupdf.FzRect_fz_new_bbox_device(self)