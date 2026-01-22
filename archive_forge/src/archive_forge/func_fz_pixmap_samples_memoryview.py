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
def fz_pixmap_samples_memoryview(pixmap):
    """
    Returns a writable Python `memoryview` for a `FzPixmap`.
    """
    return ll_fz_pixmap_samples_memoryview(pixmap.m_internal)