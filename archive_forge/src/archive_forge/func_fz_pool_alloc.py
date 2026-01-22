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
def fz_pool_alloc(self, size):
    """
        Class-aware wrapper for `::fz_pool_alloc()`.
        	Allocate a block of size bytes from the pool.
        """
    return _mupdf.FzPool_fz_pool_alloc(self, size)