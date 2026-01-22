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
def ll_fz_md5_update(state, input, inlen):
    """
    Low-level wrapper for `::fz_md5_update()`.
    MD5 block update operation. Continues an MD5 message-digest
    operation, processing another message block, and updating the
    context.

    Never throws an exception.
    """
    return _mupdf.ll_fz_md5_update(state, input, inlen)