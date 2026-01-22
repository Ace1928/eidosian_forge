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
def ll_fz_dom_add_attribute(elt, att, value):
    """
    Low-level wrapper for `::fz_dom_add_attribute()`.
    Add an attribute to an element.

    Ownership of att and value remain with the caller.
    """
    return _mupdf.ll_fz_dom_add_attribute(elt, att, value)