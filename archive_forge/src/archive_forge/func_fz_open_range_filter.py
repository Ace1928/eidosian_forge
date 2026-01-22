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
def fz_open_range_filter(self, ranges, nranges):
    """
        Class-aware wrapper for `::fz_open_range_filter()`.
        	The range filter copies data from specified ranges of the
        	chained stream.
        """
    return _mupdf.FzStream_fz_open_range_filter(self, ranges, nranges)