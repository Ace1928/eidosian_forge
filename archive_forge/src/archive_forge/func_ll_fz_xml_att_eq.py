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
def ll_fz_xml_att_eq(item, name, match):
    """
    Low-level wrapper for `::fz_xml_att_eq()`.
    Check for a matching attribute on an XML node.

    If the node has the requested attribute (name), and the value
    matches (match) then return 1. Otherwise, 0.
    """
    return _mupdf.ll_fz_xml_att_eq(item, name, match)