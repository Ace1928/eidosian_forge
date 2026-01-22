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
def fz_open_arc4(self, key, keylen):
    """
        Class-aware wrapper for `::fz_open_arc4()`.
        	arc4 filter performs RC4 decoding of data read from the chained
        	filter using the supplied key.
        """
    return _mupdf.FzStream_fz_open_arc4(self, key, keylen)