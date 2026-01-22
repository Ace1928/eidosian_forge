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
def fz_open_image_decomp_stream_outparams_fn(arg_0, arg_1):
    """
    Class-aware helper for out-params of fz_open_image_decomp_stream() [fz_open_image_decomp_stream()].
    """
    ret, l2factor = ll_fz_open_image_decomp_stream(arg_0.m_internal, arg_1.m_internal)
    return (FzStream(ret), l2factor)