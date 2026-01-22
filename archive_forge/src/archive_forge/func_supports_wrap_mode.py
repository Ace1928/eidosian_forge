from __future__ import annotations
import functools
import typing
from urwid.str_util import calc_text_pos, calc_width, get_char_width, is_wide_char, move_next_char, move_prev_char
from urwid.util import calc_trim_text, get_encoding
def supports_wrap_mode(self, wrap: Literal['any', 'space', 'clip', 'ellipsis'] | WrapMode) -> bool:
    """Return True if wrap is 'any', 'space', 'clip' or 'ellipsis'."""
    return wrap in {'any', 'space', 'clip', 'ellipsis'}