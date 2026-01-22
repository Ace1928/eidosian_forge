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
def fz_run_page_annots(self, dev, transform, cookie):
    """
        Class-aware wrapper for `::fz_run_page_annots()`.
        	Run the annotations on a page through a device.
        """
    return _mupdf.FzPage_fz_run_page_annots(self, dev, transform, cookie)