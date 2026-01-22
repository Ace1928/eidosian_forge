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
def ll_fz_walk_path(path, walker, arg):
    """
    Low-level wrapper for `::fz_walk_path()`.
    Walk the segments of a path, calling the
    appropriate callback function from a given set for each
    segment of the path.

    path: The path to walk.

    walker: The set of callback functions to use. The first
    4 callback pointers in the set must be non-NULL. The
    subsequent ones can either be supplied, or can be left
    as NULL, in which case the top 4 functions will be
    called as appropriate to simulate them.

    arg: An opaque argument passed in to each callback.

    Exceptions will only be thrown if the underlying callback
    functions throw them.
    """
    return _mupdf.ll_fz_walk_path(path, walker, arg)