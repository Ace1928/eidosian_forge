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
def fz_count_archive_entries(self):
    """
        Class-aware wrapper for `::fz_count_archive_entries()`.
        	Number of entries in archive.

        	Will always return a value >= 0.

        	May throw an exception if this type of archive cannot count the
        	entries (such as a directory).
        """
    return _mupdf.FzArchive_fz_count_archive_entries(self)