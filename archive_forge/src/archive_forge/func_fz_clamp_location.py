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
def fz_clamp_location(self, loc):
    """
        Class-aware wrapper for `::fz_clamp_location()`.
        	Clamps a location into valid chapter/page range. (First clamps
        	the chapter into range, then the page into range).
        """
    return _mupdf.FzDocument_fz_clamp_location(self, loc)