from __future__ import annotations
import functools
import typing
from urwid.str_util import calc_text_pos, calc_width, get_char_width, is_wide_char, move_next_char, move_prev_char
from urwid.util import calc_trim_text, get_encoding
def shift_line(segs: list[tuple[int, int, int | bytes] | tuple[int, int | None]], amount: int) -> list[tuple[int, int, int | bytes] | tuple[int, int | None]]:
    """
    Return a shifted line from a layout structure to the left or right.
    segs -- line of a layout structure
    amount -- screen columns to shift right (+ve) or left (-ve)
    """
    if not isinstance(amount, int):
        raise TypeError(amount)
    if segs and len(segs[0]) == 2 and (segs[0][1] is None):
        amount += segs[0][0]
        if amount:
            return [(amount, None)] + segs[1:]
        return segs[1:]
    if amount:
        return [(amount, None), *segs]
    return segs