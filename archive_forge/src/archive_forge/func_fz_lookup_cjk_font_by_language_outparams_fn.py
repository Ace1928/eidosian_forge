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
def fz_lookup_cjk_font_by_language_outparams_fn(lang):
    """
    Class-aware helper for out-params of fz_lookup_cjk_font_by_language() [fz_lookup_cjk_font_by_language()].
    """
    ret, len, subfont = ll_fz_lookup_cjk_font_by_language(lang)
    return (ret, len, subfont)