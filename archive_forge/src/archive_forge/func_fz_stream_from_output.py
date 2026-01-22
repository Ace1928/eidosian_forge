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
def fz_stream_from_output(self):
    """
        Class-aware wrapper for `::fz_stream_from_output()`.
        	Obtain the fz_output in the form of a fz_stream.

        	This allows data to be read back from some forms of fz_output
        	object. When finished reading, the fz_stream should be released
        	by calling fz_drop_stream. Until the fz_stream is dropped, no
        	further operations should be performed on the fz_output object.
        """
    return _mupdf.FzOutput_fz_stream_from_output(self)