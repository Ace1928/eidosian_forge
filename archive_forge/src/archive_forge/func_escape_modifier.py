from __future__ import annotations
import re
import sys
import typing
from collections.abc import MutableMapping, Sequence
from urwid import str_util
import urwid.util  # isort: skip  # pylint: disable=wrong-import-position
def escape_modifier(digit: str) -> str:
    mode = ord(digit) - ord('1')
    return 'shift ' * (mode & 1) + 'meta ' * ((mode & 2) // 2) + 'ctrl ' * ((mode & 4) // 4)