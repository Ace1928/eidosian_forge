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
def ll_fz_drop_archive(arch):
    """
    Low-level wrapper for `::fz_drop_archive()`.
    Drop a reference to an archive.

    When the last reference is dropped, this closes and releases
    any memory or filehandles associated with the archive.
    """
    return _mupdf.ll_fz_drop_archive(arch)