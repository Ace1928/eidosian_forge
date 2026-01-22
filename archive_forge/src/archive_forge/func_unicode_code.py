import re
import codecs
import sys
from docutils import nodes
from docutils.utils import split_escaped_whitespace, escape2null, unescape
from docutils.parsers.rst.languages import en as _fallback_language_module
def unicode_code(code):
    """
    Convert a Unicode character code to a Unicode character.
    (Directive option conversion function.)

    Codes may be decimal numbers, hexadecimal numbers (prefixed by ``0x``,
    ``x``, ``\\x``, ``U+``, ``u``, or ``\\u``; e.g. ``U+262E``), or XML-style
    numeric character entities (e.g. ``&#x262E;``).  Other text remains as-is.

    Raise ValueError for illegal Unicode code values.
    """
    try:
        if code.isdigit():
            return chr(int(code))
        else:
            match = unicode_pattern.match(code)
            if match:
                value = match.group(1) or match.group(2)
                return chr(int(value, 16))
            else:
                return code
    except OverflowError as detail:
        raise ValueError('code too large (%s)' % detail)