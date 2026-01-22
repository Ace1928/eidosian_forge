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
def fz_open_directory(path):
    """
    Class-aware wrapper for `::fz_open_directory()`.
    	Open a directory as if it was an archive.

    	A special case where a directory is opened as if it was an
    	archive.

    	Note that for directories it is not possible to retrieve the
    	number of entries or list the entries. It is however possible
    	to check if the archive has a particular entry.

    	path: a path to a directory as it would be given to opendir(3).
    """
    return _mupdf.fz_open_directory(path)