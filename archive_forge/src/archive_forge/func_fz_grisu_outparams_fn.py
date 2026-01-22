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
def fz_grisu_outparams_fn(f, s):
    """
    Class-aware helper for out-params of fz_grisu() [fz_grisu()].
    """
    ret, exp = ll_fz_grisu(f, s)
    return (ret, exp)