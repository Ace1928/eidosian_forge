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
def fz_truncate_output(self):
    """
        Class-aware wrapper for `::fz_truncate_output()`.
        	Truncate the output at the current position.

        	This allows output streams which have seeked back from the end
        	of their storage to be truncated at the current point.
        """
    return _mupdf.FzOutput_fz_truncate_output(self)