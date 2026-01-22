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
def ll_fz_bidi_fragment_text(text, textlen, callback, arg, flags):
    """
    Wrapper for out-params of fz_bidi_fragment_text().
    Returns: ::fz_bidi_direction baseDir
    """
    outparams = ll_fz_bidi_fragment_text_outparams()
    ret = ll_fz_bidi_fragment_text_outparams_fn(text, textlen, callback, arg, flags, outparams)
    return outparams.baseDir