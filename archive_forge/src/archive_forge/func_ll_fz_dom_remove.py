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
def ll_fz_dom_remove(elt):
    """
    Low-level wrapper for `::fz_dom_remove()`.
    Remove an element from the DOM. The element can be added back elsewhere
    if required.

    No reference counting changes for the element.
    """
    return _mupdf.ll_fz_dom_remove(elt)