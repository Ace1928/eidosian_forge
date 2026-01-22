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
def shards_trim_top(shards, top: int):
    """
    Return shards with top rows removed.
    """
    if top <= 0:
        raise ValueError(top)
    shard_iter = iter(shards)
    shard_tail = []
    for num_rows, cviews in shard_iter:
        if top < num_rows:
            break
        sbody = shard_body(cviews, shard_tail, False)
        shard_tail = shard_body_tail(num_rows, sbody)
        top -= num_rows
    else:
        raise CanvasError('tried to trim shards out of existence')
    sbody = shard_body(cviews, shard_tail, False)
    shard_tail = shard_body_tail(num_rows, sbody)
    new_sbody = [(0, content_iter, cview_trim_top(cv, done_rows + top)) for done_rows, content_iter, cv in sbody]
    sbody = new_sbody
    new_shards = [(num_rows - top, [cv for done_rows, content_iter, cv in sbody])]
    new_shards.extend(shard_iter)
    return new_shards