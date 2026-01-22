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
def fz_glyph_cacheable(self, gid):
    """
        Class-aware wrapper for `::fz_glyph_cacheable()`.
        	Determine if a given glyph in a font
        	is cacheable. Certain glyphs in a type 3 font cannot safely
        	be cached, as their appearance depends on the enclosing
        	graphic state.

        	font: The font to look for the glyph in.

        	gif: The glyph to query.

        	Returns non-zero if cacheable, 0 if not.
        """
    return _mupdf.FzFont_fz_glyph_cacheable(self, gid)