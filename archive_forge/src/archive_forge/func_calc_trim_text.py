from __future__ import annotations
import codecs
import contextlib
import sys
import typing
import warnings
from contextlib import suppress
from urwid import str_util
def calc_trim_text(text: str | bytes, start_offs: int, end_offs: int, start_col: int, end_col: int) -> tuple[int, int, int, int]:
    """
    Calculate the result of trimming text.
    start_offs -- offset into text to treat as screen column 0
    end_offs -- offset into text to treat as the end of the line
    start_col -- screen column to trim at the left
    end_col -- screen column to trim at the right

    Returns (start, end, pad_left, pad_right), where:
    start -- resulting start offset
    end -- resulting end offset
    pad_left -- 0 for no pad or 1 for one space to be added
    pad_right -- 0 for no pad or 1 for one space to be added
    """
    spos = start_offs
    pad_left = pad_right = 0
    if start_col > 0:
        spos, sc = str_util.calc_text_pos(text, spos, end_offs, start_col)
        if sc < start_col:
            pad_left = 1
            spos, sc = str_util.calc_text_pos(text, start_offs, end_offs, start_col + 1)
    run = end_col - start_col - pad_left
    pos, sc = str_util.calc_text_pos(text, spos, end_offs, run)
    if sc < run:
        pad_right = 1
    return (spos, pos, pad_left, pad_right)