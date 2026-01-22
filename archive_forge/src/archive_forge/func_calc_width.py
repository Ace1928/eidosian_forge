from __future__ import annotations
import re
import typing
import warnings
import wcwidth
def calc_width(text: str | bytes, start_offs: int, end_offs: int) -> int:
    """
    Return the screen column width of text between start_offs and end_offs.

    text may be unicode or a byte string in the target _byte_encoding

    Some characters are wide (take two columns) and others affect the
    previous character (take zero columns).  Use the widths table above
    to calculate the screen column width of text[start_offs:end_offs]
    """
    if start_offs > end_offs:
        raise ValueError((start_offs, end_offs))
    if isinstance(text, str):
        return sum((get_char_width(char) for char in text[start_offs:end_offs]))
    if _byte_encoding == 'utf8':
        try:
            return sum((get_char_width(char) for char in text[start_offs:end_offs].decode('utf-8')))
        except UnicodeDecodeError as exc:
            warnings.warn(f'`calc_width` with text encoded to bytes can produce incorrect resultsdue to possible offset in the middle of character: {exc}', UnicodeWarning, stacklevel=2)
        i = start_offs
        sc = 0
        while i < end_offs:
            o, i = decode_one(text, i)
            w = get_width(o)
            sc += w
        return sc
    return end_offs - start_offs