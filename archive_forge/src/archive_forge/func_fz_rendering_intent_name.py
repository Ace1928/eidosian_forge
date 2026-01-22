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
def fz_rendering_intent_name(ri):
    """
    Class-aware wrapper for `::fz_rendering_intent_name()`.
    	Map from enumerated rendering intent to string.

    	The returned string is static and therefore must not be freed.
    """
    return _mupdf.fz_rendering_intent_name(ri)