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
def mouse_event(self, size, event, button, col, row, focus):
    if not hasattr(self.bottom_w, 'mouse_event'):
        return False
    return self.bottom_w.mouse_event(size, event, button, col, row, focus)