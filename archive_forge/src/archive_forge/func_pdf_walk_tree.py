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
def pdf_walk_tree(self, kid_name, arrive, leave, arg, names, values):
    """
        Class-aware wrapper for `::pdf_walk_tree()`.

        This method has out-params. Python/C# wrappers look like:
        	`pdf_walk_tree(::pdf_obj *kid_name, void (*arrive)(::fz_context *, ::pdf_obj *, void *, ::pdf_obj **), void (*leave)(::fz_context *, ::pdf_obj *, void *), void *arg, ::pdf_obj **names, ::pdf_obj **values)` => `()`
        """
    return _mupdf.PdfObj_pdf_walk_tree(self, kid_name, arrive, leave, arg, names, values)