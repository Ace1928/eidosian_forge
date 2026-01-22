from __future__ import annotations
import re
import typing
import warnings
import wcwidth
def within_double_byte(text: bytes, line_start: int, pos: int) -> Literal[0, 1, 2]:
    """Return whether pos is within a double-byte encoded character.

    text -- byte string in question
    line_start -- offset of beginning of line (< pos)
    pos -- offset in question

    Return values:
    0 -- not within dbe char, or double_byte_encoding == False
    1 -- pos is on the 1st half of a dbe char
    2 -- pos is on the 2nd half of a dbe char
    """
    if not isinstance(text, bytes):
        raise TypeError(text)
    v = text[pos]
    if 64 <= v < 127:
        if pos == line_start:
            return 0
        if text[pos - 1] >= 129 and within_double_byte(text, line_start, pos - 1) == 1:
            return 2
        return 0
    if v < 128:
        return 0
    i = pos - 1
    while i >= line_start:
        if text[i] < 128:
            break
        i -= 1
    if pos - i & 1:
        return 1
    return 2