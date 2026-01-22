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
def fz_new_test_device_outparams_fn(threshold, options, passthrough):
    """
    Class-aware helper for out-params of fz_new_test_device() [fz_new_test_device()].
    """
    ret, is_color = ll_fz_new_test_device(threshold, options, passthrough.m_internal)
    return (FzDevice(ret), is_color)