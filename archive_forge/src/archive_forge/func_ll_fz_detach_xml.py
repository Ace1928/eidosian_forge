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
def ll_fz_detach_xml(node):
    """
    Low-level wrapper for `::fz_detach_xml()`.
    Detach a node from the tree, unlinking it from its parent,
    and setting the document root to the node.
    """
    return _mupdf.ll_fz_detach_xml(node)