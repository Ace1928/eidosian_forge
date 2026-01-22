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
def fz_is_external_link(uri):
    """
    Class-aware wrapper for `::fz_is_external_link()`.
    	Query whether a link is external to a document (determined by
    	uri containing a ':', intended to match with '://' which
    	separates the scheme from the scheme specific parts in URIs).
    """
    return _mupdf.fz_is_external_link(uri)