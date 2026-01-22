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
def fz_set_stddbg(self):
    """
        Class-aware wrapper for `::fz_set_stddbg()`.
        	Set the output stream to be used for fz_stddbg. Set to NULL to
        	reset to default (stderr).
        """
    return _mupdf.FzOutput_fz_set_stddbg(self)