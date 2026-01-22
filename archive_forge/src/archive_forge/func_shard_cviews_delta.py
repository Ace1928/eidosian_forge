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
def shard_cviews_delta(cviews, other_cviews):
    other_cviews_iter = iter(other_cviews)
    other_cv = None
    cols = other_cols = 0
    for cv in cviews:
        if other_cv is None:
            other_cv = next(other_cviews_iter)
        while other_cols < cols:
            other_cols += other_cv[2]
            other_cv = next(other_cviews_iter)
        if other_cols > cols:
            yield cv
            cols += cv[2]
            continue
        if cv[5] is other_cv[5] and cv[:5] == other_cv[:5]:
            yield (cv[:5] + (None,) + cv[6:])
        else:
            yield cv
        other_cols += other_cv[2]
        other_cv = None
        cols += cv[2]