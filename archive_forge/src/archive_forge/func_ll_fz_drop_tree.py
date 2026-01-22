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
def ll_fz_drop_tree(node, dropfunc):
    """
    Low-level wrapper for `::fz_drop_tree()`.
    Drop the tree.

    The storage used by the tree is freed, and each value has
    dropfunc called on it.
    """
    return _mupdf.ll_fz_drop_tree(node, dropfunc)