from __future__ import annotations
import contextlib
import dataclasses
import typing
import warnings
import weakref
from contextlib import suppress
from urwid.str_util import calc_text_pos, calc_width
from urwid.text_layout import LayoutSegment, trim_line
from urwid.util import (
def translated_coords(self, dx: int, dy: int) -> tuple[int, int] | None:
    """
        Return cursor coords shifted by (dx, dy), or None if there
        is no cursor.
        """
    if self.cursor:
        x, y = self.cursor
        return (x + dx, y + dy)
    return None