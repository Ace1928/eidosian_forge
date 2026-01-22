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
def fz_open_endstream_filter(self, len, offset):
    """
        Class-aware wrapper for `::fz_open_endstream_filter()`.
        	The endstream filter reads a PDF substream, and starts to look
        	for an 'endstream' token after the specified length.
        """
    return _mupdf.FzStream_fz_open_endstream_filter(self, len, offset)