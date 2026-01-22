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
def ll_fz_location_from_page_number(doc, number):
    """
    Low-level wrapper for `::fz_location_from_page_number()`.
    Converts from page number to chapter+page. This may cause many
    chapters to be laid out in order to calculate the number of
    pages within those chapters.
    """
    return _mupdf.ll_fz_location_from_page_number(doc, number)