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
def fz_seek_output(self, off, whence):
    """
        Class-aware wrapper for `::fz_seek_output()`.
        	Seek to the specified file position.
        	See fseek for arguments.

        	Throw an error on unseekable outputs.
        """
    return _mupdf.FzOutput_fz_seek_output(self, off, whence)