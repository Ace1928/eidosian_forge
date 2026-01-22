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
def fz_new_deflated_data_from_buffer_outparams_fn(buffer, level):
    """
    Class-aware helper for out-params of fz_new_deflated_data_from_buffer() [fz_new_deflated_data_from_buffer()].
    """
    ret, compressed_length = ll_fz_new_deflated_data_from_buffer(buffer.m_internal, level)
    return (ret, compressed_length)