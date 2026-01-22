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
def pad_trim_top_bottom(self, top: int, bottom: int) -> None:
    """
        Pad or trim this canvas on the top and bottom.
        """
    if self.widget_info:
        raise self._finalized_error
    orig_shards = self.shards
    if top < 0 or bottom < 0:
        trim_top = max(0, -top)
        rows = self.rows() - trim_top - max(0, -bottom)
        self.trim(trim_top, rows)
    cols = self.cols()
    if top > 0:
        self.shards = [(top, [(0, 0, cols, top, None, blank_canvas)]), *self.shards]
        self.coords = self.translate_coords(0, top)
    if bottom > 0:
        if orig_shards is self.shards:
            self.shards = self.shards[:]
        self.shards.append((bottom, [(0, 0, cols, bottom, None, blank_canvas)]))