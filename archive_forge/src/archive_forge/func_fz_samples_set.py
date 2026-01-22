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
def fz_samples_set(self, offset, value):
    """
        Class-aware wrapper for `::fz_samples_set()`.
        Provides simple (but slow) write access to pixmap data from Python and
        C#.
        """
    return _mupdf.FzPixmap_fz_samples_set(self, offset, value)