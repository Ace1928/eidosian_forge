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
def ll_fz_device_current_scissor(dev):
    """
    Low-level wrapper for `::fz_device_current_scissor()`.
    Find current scissor region as tracked by the device.
    """
    return _mupdf.ll_fz_device_current_scissor(dev)