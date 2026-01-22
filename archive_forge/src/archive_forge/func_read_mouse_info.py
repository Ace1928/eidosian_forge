from __future__ import annotations
import re
import sys
import typing
from collections.abc import MutableMapping, Sequence
from urwid import str_util
import urwid.util  # isort: skip  # pylint: disable=wrong-import-position
def read_mouse_info(self, keys: Collection[int], more_available: bool):
    if len(keys) < 3:
        if more_available:
            raise MoreInputRequired()
        return None
    b = keys[0] - 32
    x, y = ((keys[1] - 33) % 256, (keys[2] - 33) % 256)
    prefixes = []
    if b & 4:
        prefixes.append('shift ')
    if b & 8:
        prefixes.append('meta ')
    if b & 16:
        prefixes.append('ctrl ')
    if (b & MOUSE_MULTIPLE_CLICK_MASK) >> 9 == 1:
        prefixes.append('double ')
    if (b & MOUSE_MULTIPLE_CLICK_MASK) >> 9 == 2:
        prefixes.append('triple ')
    prefix = ''.join(prefixes)
    button = (b & 64) // 64 * 3 + (b & 3) + 1
    if b & 3 == 3:
        action = 'release'
        button = 0
    elif b & MOUSE_RELEASE_FLAG:
        action = 'release'
    elif b & MOUSE_DRAG_FLAG:
        action = 'drag'
    elif b & MOUSE_MULTIPLE_CLICK_MASK:
        action = 'click'
    else:
        action = 'press'
    return ((f'{prefix}mouse {action}', button, x, y), keys[3:])