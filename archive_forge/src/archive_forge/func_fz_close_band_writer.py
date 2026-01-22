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
def fz_close_band_writer(self):
    """
        Class-aware wrapper for `::fz_close_band_writer()`.
        	Finishes up the output and closes the band writer. After this
        	call no more headers or bands may be written.
        """
    return _mupdf.FzBandWriter_fz_close_band_writer(self)