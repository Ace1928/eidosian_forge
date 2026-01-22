import sys
import os
import time
import locale
import signal
import urwid
from typing import Optional
from . import args as bpargs, repl, translations
from .formatter import theme_map
from .translations import _
from .keys import urwid_key_dispatch as key_dispatch
def move_cursor_to_coords(self, *args):
    if self._bpy_may_move_cursor:
        return super().move_cursor_to_coords(*args)
    return False