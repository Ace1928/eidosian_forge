import sys
import encodings
import encodings.aliases
import re
import _collections_abc
from builtins import str as _builtin_str
import functools
def delocalize(string):
    """Parses a string as a normalized number according to the locale settings."""
    conv = localeconv()
    ts = conv['thousands_sep']
    if ts:
        string = string.replace(ts, '')
    dd = conv['decimal_point']
    if dd:
        string = string.replace(dd, '.')
    return string