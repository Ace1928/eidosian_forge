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
def ll_fz_new_buffer_from_copied_data(data):
    """
    Returns fz_buffer containing copy of `data`, which should
    be a `bytes` or similar Python buffer instance.
    """
    buffer_ = ll_fz_new_buffer_from_copied_data_orig(python_buffer_data(data), len(data))
    return buffer_