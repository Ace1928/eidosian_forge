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
def fz_encode_character_with_fallback(self, unicode, script, language, out_font):
    """
        Class-aware wrapper for `::fz_encode_character_with_fallback()`.

        This method has out-params. Python/C# wrappers look like:
        	`fz_encode_character_with_fallback(int unicode, int script, int language, ::fz_font **out_font)` => `(int)`

        	Find the glyph id for
        	a given unicode character within a font, falling back to
        	an alternative if not found.

        	font: The font to look for the unicode character in.

        	unicode: The unicode character to encode.

        	script: The script in use.

        	language: The language in use.

        	out_font: The font handle in which the given glyph represents
        	the requested unicode character. The caller does not own the
        	reference it is passed, so should call fz_keep_font if it is
        	not simply to be used immediately.

        	Returns the glyph id for the given unicode value in the supplied
        	font (and sets *out_font to font) if it is present. Otherwise
        	an alternative fallback font (based on script/language) is
        	searched for. If the glyph is found therein, *out_font is set
        	to this reference, and the glyph reference is returned. If it
        	cannot be found anywhere, the function returns 0.
        """
    return _mupdf.FzFont_fz_encode_character_with_fallback(self, unicode, script, language, out_font)