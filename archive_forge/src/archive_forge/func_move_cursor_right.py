from __future__ import annotations
import re
import sys
import typing
from collections.abc import MutableMapping, Sequence
from urwid import str_util
import urwid.util  # isort: skip  # pylint: disable=wrong-import-position
def move_cursor_right(x: int) -> str:
    if x < 1:
        return ''
    return ESC + f'[{x:d}C'