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
def shard_body_row(sbody):
    """
    Return one row, advancing the iterators in sbody.

    ** MODIFIES sbody by calling next() on its iterators **
    """
    row = []
    for _done_rows, content_iter, cview in sbody:
        if content_iter:
            row.extend(next(content_iter))
        elif row and isinstance(row[-1], int):
            row[-1] = row[-1] + cview[2]
        else:
            row.append(cview[2])
    return row