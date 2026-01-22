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
def fz_add_separation(self, name, cs, cs_channel):
    """
        Class-aware wrapper for `::fz_add_separation()`.
        	Add a separation (null terminated name, colorspace)
        """
    return _mupdf.FzSeparations_fz_add_separation(self, name, cs, cs_channel)