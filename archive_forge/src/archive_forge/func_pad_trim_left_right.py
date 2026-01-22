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
def pad_trim_left_right(self, left: int, right: int) -> None:
    """
        Pad or trim this canvas on the left and right

        values > 0 indicate screen columns to pad
        values < 0 indicate screen columns to trim
        """
    if self.widget_info:
        raise self._finalized_error
    shards = self.shards
    if left < 0 or right < 0:
        trim_left = max(0, -left)
        cols = self.cols() - trim_left - max(0, -right)
        shards = shards_trim_sides(shards, trim_left, cols)
    rows = self.rows()
    if left > 0 or right > 0:
        top_rows, top_cviews = shards[0]
        if left > 0:
            new_top_cviews = [(0, 0, left, rows, None, blank_canvas), *top_cviews]
        else:
            new_top_cviews = top_cviews.copy()
        if right > 0:
            new_top_cviews.append((0, 0, right, rows, None, blank_canvas))
        shards = [(top_rows, new_top_cviews)] + shards[1:]
    self.coords = self.translate_coords(left, 0)
    self.shards = shards