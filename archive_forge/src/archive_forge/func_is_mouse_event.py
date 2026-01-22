from __future__ import annotations
import codecs
import contextlib
import sys
import typing
import warnings
from contextlib import suppress
from urwid import str_util
def is_mouse_event(ev: tuple[str, int, int, int] | typing.Any) -> bool:
    return isinstance(ev, tuple) and len(ev) == 4 and ('mouse' in ev[0])