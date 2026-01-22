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
def ll_fz_pixmap_samples_memoryview(pixmap):
    """
    Returns a writable Python `memoryview` for a `fz_pixmap`.
    """
    assert isinstance(pixmap, fz_pixmap)
    ret = python_memoryview_from_memory(ll_fz_pixmap_samples(pixmap), ll_fz_pixmap_stride(pixmap) * ll_fz_pixmap_height(pixmap), 1)
    return ret