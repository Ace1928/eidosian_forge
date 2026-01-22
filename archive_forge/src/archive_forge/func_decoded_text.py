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
@property
def decoded_text(self) -> Sequence[str]:
    """Decoded text content of the canvas as a sequence of strings, one for each row."""
    encoding = get_encoding()
    return tuple((line.decode(encoding) for line in self.text))