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
def fz_archive_format(self):
    """
        Class-aware wrapper for `::fz_archive_format()`.
        	Return a pointer to a string describing the format of the
        	archive.

        	The lifetime of the string is unspecified (in current
        	implementations the string will persist until the archive
        	is closed, but this is not guaranteed).
        """
    return _mupdf.FzArchive_fz_archive_format(self)