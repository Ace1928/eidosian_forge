from __future__ import annotations
import re
import typing
import warnings
import wcwidth
def calc_text_pos(text: str | bytes, start_offs: int, end_offs: int, pref_col: int) -> tuple[int, int]:
    """
    Calculate the closest position to the screen column pref_col in text
    where start_offs is the offset into text assumed to be screen column 0
    and end_offs is the end of the range to search.

    text may be unicode or a byte string in the target _byte_encoding

    Returns (position, actual_col).
    """
    if start_offs > end_offs:
        raise ValueError((start_offs, end_offs))
    if isinstance(text, str):
        return calc_string_text_pos(text, start_offs, end_offs, pref_col)
    if not isinstance(text, bytes):
        raise TypeError(text)
    if _byte_encoding == 'utf8':
        i = start_offs
        sc = 0
        while i < end_offs:
            o, n = decode_one(text, i)
            w = get_width(o)
            if w + sc > pref_col:
                return (i, sc)
            i = n
            sc += w
        return (i, sc)
    i = start_offs + pref_col
    if i >= end_offs:
        return (end_offs, end_offs - start_offs)
    if _byte_encoding == 'wide' and within_double_byte(text, start_offs, i) == 2:
        i -= 1
    return (i, i - start_offs)