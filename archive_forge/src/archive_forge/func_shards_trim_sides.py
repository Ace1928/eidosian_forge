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
def shards_trim_sides(shards, left: int, cols: int):
    """
    Return shards with starting from column left and cols total width.
    """
    if left < 0:
        raise ValueError(left)
    if cols <= 0:
        raise ValueError(cols)
    shard_tail = []
    new_shards = []
    right = left + cols
    for num_rows, cviews in shards:
        sbody = shard_body(cviews, shard_tail, False)
        shard_tail = shard_body_tail(num_rows, sbody)
        new_cviews = []
        col = 0
        for done_rows, _content_iter, cv in sbody:
            cv_cols = cv[2]
            next_col = col + cv_cols
            if done_rows or next_col <= left or col >= right:
                col = next_col
                continue
            if col < left:
                cv = cview_trim_left(cv, left - col)
                col = left
            if next_col > right:
                cv = cview_trim_cols(cv, right - col)
            new_cviews.append(cv)
            col = next_col
        if not new_cviews:
            prev_num_rows, prev_cviews = new_shards[-1]
            new_shards[-1] = (prev_num_rows + num_rows, prev_cviews)
        else:
            new_shards.append((num_rows, new_cviews))
    return new_shards