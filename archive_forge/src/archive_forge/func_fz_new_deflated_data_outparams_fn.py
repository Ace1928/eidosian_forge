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
def fz_new_deflated_data_outparams_fn(source, source_length, level):
    """
    Class-aware helper for out-params of fz_new_deflated_data() [fz_new_deflated_data()].
    """
    ret, compressed_length = ll_fz_new_deflated_data(source, source_length, level)
    return (ret, compressed_length)