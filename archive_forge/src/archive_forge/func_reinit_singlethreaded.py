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
def reinit_singlethreaded():
    """
     Reinitializes the MuPDF context for single-threaded use, which
    is slightly faster when calling code is single threaded.

    This should be called before any other use of MuPDF.
    """
    return _mupdf.reinit_singlethreaded()