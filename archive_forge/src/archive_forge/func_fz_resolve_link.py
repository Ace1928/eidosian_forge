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
def fz_resolve_link(self, uri, xp, yp):
    """
        Class-aware wrapper for `::fz_resolve_link()`.

        This method has out-params. Python/C# wrappers look like:
        	`fz_resolve_link(const char *uri)` => `(fz_location, float xp, float yp)`

        	Resolve an internal link to a page number.

        	xp, yp: Pointer to store coordinate of destination on the page.

        	Returns (-1,-1) if the URI cannot be resolved.
        """
    return _mupdf.FzDocument_fz_resolve_link(self, uri, xp, yp)