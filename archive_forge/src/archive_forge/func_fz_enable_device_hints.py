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
def fz_enable_device_hints(self, hints):
    """
        Class-aware wrapper for `::fz_enable_device_hints()`.
        	Enable (set) hint bits within the hint bitfield for a device.
        """
    return _mupdf.FzDevice_fz_enable_device_hints(self, hints)