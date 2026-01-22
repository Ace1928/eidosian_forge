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
def fz_search_display_list(self, needle, hit_mark, hit_bbox, hit_max):
    """
        Class-aware wrapper for `::fz_search_display_list()`.

        This method has out-params. Python/C# wrappers look like:
        	`fz_search_display_list(const char *needle, ::fz_quad *hit_bbox, int hit_max)` => `(int, int hit_mark)`
        """
    return _mupdf.FzDisplayList_fz_search_display_list(self, needle, hit_mark, hit_bbox, hit_max)