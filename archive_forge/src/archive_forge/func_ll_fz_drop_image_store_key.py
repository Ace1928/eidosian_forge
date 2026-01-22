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
def ll_fz_drop_image_store_key(image):
    """
    Low-level wrapper for `::fz_drop_image_store_key()`.
    Decrement the store key reference count for an image. When the
    total (normal + key) reference count reaches zero, the image and
    its resources are freed.

    Never throws exceptions.
    """
    return _mupdf.ll_fz_drop_image_store_key(image)