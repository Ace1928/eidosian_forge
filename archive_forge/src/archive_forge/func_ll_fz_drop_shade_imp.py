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
def ll_fz_drop_shade_imp(shade):
    """
    Low-level wrapper for `::fz_drop_shade_imp()`.
    Internal function to destroy a
    shade. Only exposed for use with the fz_store.

    shade: The reference to destroy.
    """
    return _mupdf.ll_fz_drop_shade_imp(shade)