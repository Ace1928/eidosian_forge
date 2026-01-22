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
def fz_open_cfb_archive(filename):
    """
    Class-aware wrapper for `::fz_open_cfb_archive()`.
    	Open a cfb file as an archive.

    	An exception is thrown if the file is not recognised as a cfb.

    	filename: a path to an archive file as it would be given to
    	open(2).
    """
    return _mupdf.fz_open_cfb_archive(filename)