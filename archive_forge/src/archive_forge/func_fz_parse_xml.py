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
def fz_parse_xml(self, preserve_white):
    """
        Class-aware wrapper for `::fz_parse_xml()`.
        	Parse the contents of buffer into a tree of xml nodes.

        	preserve_white: whether to keep or delete all-whitespace nodes.
        """
    return _mupdf.FzBuffer_fz_parse_xml(self, preserve_white)