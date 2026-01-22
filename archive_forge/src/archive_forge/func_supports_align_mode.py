from __future__ import annotations
import functools
import typing
from urwid.str_util import calc_text_pos, calc_width, get_char_width, is_wide_char, move_next_char, move_prev_char
from urwid.util import calc_trim_text, get_encoding
def supports_align_mode(self, align: Literal['left', 'center', 'right'] | Align) -> bool:
    """Return True if align is 'left', 'center' or 'right'."""
    return align in {'left', 'center', 'right'}