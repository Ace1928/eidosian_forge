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
def ll_fz_new_device_of_size(size):
    """
    Low-level wrapper for `::fz_new_device_of_size()`.
    Devices are created by calls to device implementations, for
    instance: foo_new_device(). These will be implemented by calling
    fz_new_derived_device(ctx, foo_device) where foo_device is a
    structure "derived from" fz_device, for instance
    typedef struct { fz_device base;  ...extras...} foo_device;
    """
    return _mupdf.ll_fz_new_device_of_size(size)