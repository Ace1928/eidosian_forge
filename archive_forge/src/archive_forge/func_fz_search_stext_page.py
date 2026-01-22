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
def fz_search_stext_page(self, needle, hit_mark, hit_bbox, hit_max):
    """
        Class-aware wrapper for `::fz_search_stext_page()`.

        This method has out-params. Python/C# wrappers look like:
        	`fz_search_stext_page(const char *needle, ::fz_quad *hit_bbox, int hit_max)` => `(int, int hit_mark)`

        	Search for occurrence of 'needle' in text page.

        	Return the number of hits and store hit quads in the passed in
        	array.

        	NOTE: This is an experimental interface and subject to change
        	without notice.
        """
    return _mupdf.FzStextPage_fz_search_stext_page(self, needle, hit_mark, hit_bbox, hit_max)