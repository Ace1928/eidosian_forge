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
def ll_fz_lookup_cjk_ordering_by_language(name):
    """
    Low-level wrapper for `::fz_lookup_cjk_ordering_by_language()`.
    Return the matching FZ_ADOBE_* ordering
    for the given language tag, such as "zh-Hant", "zh-Hans", "ja", or "ko".
    """
    return _mupdf.ll_fz_lookup_cjk_ordering_by_language(name)