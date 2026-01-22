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
def fz_getopt(nargc, nargv, ostr):
    """
    Class-aware wrapper for `::fz_getopt()`.

    This function has out-params. Python/C# wrappers look like:
    	`fz_getopt(int nargc, const char *ostr)` => `(int, char *nargv)`

    	Identical to fz_getopt_long, but with a NULL longopts field, signifying no long
    	options.
    """
    return _mupdf.fz_getopt(nargc, nargv, ostr)