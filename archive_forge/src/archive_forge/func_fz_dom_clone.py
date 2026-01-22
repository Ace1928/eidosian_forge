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
def fz_dom_clone(self):
    """
        Class-aware wrapper for `::fz_dom_clone()`.
        	Clone an element (and its children).

        	A borrowed reference to the clone is returned. The clone is not
        	yet linked into the DOM.
        """
    return _mupdf.FzXml_fz_dom_clone(self)