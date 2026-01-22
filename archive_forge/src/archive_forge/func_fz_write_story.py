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
def fz_write_story(self, story, rectfn, rectfn_ref, positionfn, positionfn_ref, pagefn, pagefn_ref):
    """ Class-aware wrapper for `::fz_write_story()`."""
    return _mupdf.FzDocumentWriter_fz_write_story(self, story, rectfn, rectfn_ref, positionfn, positionfn_ref, pagefn, pagefn_ref)