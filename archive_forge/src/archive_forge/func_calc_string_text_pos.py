from __future__ import annotations
import re
import typing
import warnings
import wcwidth
def calc_string_text_pos(text: str, start_offs: int, end_offs: int, pref_col: int) -> tuple[int, int]:
    """
    Calculate the closest position to the screen column pref_col in text
    where start_offs is the offset into text assumed to be screen column 0
    and end_offs is the end of the range to search.

    :param text: string
    :param start_offs: starting text position
    :param end_offs: ending text position
    :param pref_col: target column
    :returns: (position, actual_col)

    ..note:: this method is a simplified version of `wcwidth.wcswidth` and ideally should be in wcwidth package.
    """
    if start_offs > end_offs:
        raise ValueError((start_offs, end_offs))
    cols = 0
    for idx in range(start_offs, end_offs):
        width = get_char_width(text[idx])
        if width + cols > pref_col:
            return (idx, cols)
        cols += width
    return (end_offs, cols)