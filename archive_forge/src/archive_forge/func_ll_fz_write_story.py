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
def ll_fz_write_story(writer, story, rectfn, rectfn_ref, positionfn, positionfn_ref, pagefn, pagefn_ref):
    """ Low-level wrapper for `::fz_write_story()`."""
    return _mupdf.ll_fz_write_story(writer, story, rectfn, rectfn_ref, positionfn, positionfn_ref, pagefn, pagefn_ref)