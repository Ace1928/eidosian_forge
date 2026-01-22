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
def fz_write_data(self, data, size):
    """
        Class-aware wrapper for `::fz_write_data()`.
        	Write data to output.

        	data: Pointer to data to write.
        	size: Size of data to write in bytes.
        """
    return _mupdf.FzOutput_fz_write_data(self, data, size)