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
def maybe_redraw(loop, self):
    if self._redraw_pending:
        loop.draw_screen()
        self._redraw_pending = False
    self._redraw_handle = None