from __future__ import annotations
from fontTools.misc.textTools import byteord, tostr
import re
from bisect import bisect_right
from typing import Literal, TypeVar, overload
from . import Blocks, Scripts, ScriptExtensions, OTTags
def script_extension(char):
    """Return the script extension property assigned to the Unicode character
    'char' as a set of string.

    >>> script_extension("a") == {'Latn'}
    True
    >>> script_extension(chr(0x060C)) == {'Rohg', 'Syrc', 'Yezi', 'Arab', 'Thaa', 'Nkoo'}
    True
    >>> script_extension(chr(0x10FFFF)) == {'Zzzz'}
    True
    """
    code = byteord(char)
    i = bisect_right(ScriptExtensions.RANGES, code)
    value = ScriptExtensions.VALUES[i - 1]
    if value is None:
        return {script(char)}
    return value