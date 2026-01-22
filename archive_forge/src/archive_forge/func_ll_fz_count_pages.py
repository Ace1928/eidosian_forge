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
def ll_fz_count_pages(doc):
    """
    Low-level wrapper for `::fz_count_pages()`.
    Return the number of pages in document

    May return 0 for documents with no pages.
    """
    return _mupdf.ll_fz_count_pages(doc)