from __future__ import annotations
import functools
import typing
from urwid.str_util import calc_text_pos, calc_width, get_char_width, is_wide_char, move_next_char, move_prev_char
from urwid.util import calc_trim_text, get_encoding
def trim_line(segs: list[tuple[int, int, int | bytes] | tuple[int, int | None]], text: str | bytes, start: int, end: int) -> list[tuple[int, int, int | bytes] | tuple[int, int | None]]:
    """
    Return a trimmed line of a text layout structure.
    text -- text to which this layout structure applies
    start -- starting screen column
    end -- ending screen column
    """
    result = []
    x = 0
    for seg in segs:
        sc = seg[0]
        if start or sc < 0:
            if start >= sc:
                start -= sc
                x += sc
                continue
            s = LayoutSegment(seg)
            if x + sc >= end:
                return s.subseg(text, start, end - x)
            result += s.subseg(text, start, sc)
            start = 0
            x += sc
            continue
        if x >= end:
            break
        if x + sc > end:
            s = LayoutSegment(seg)
            result += s.subseg(text, 0, end - x)
            break
        result.append(seg)
    return result