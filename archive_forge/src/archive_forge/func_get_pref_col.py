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
def get_pref_col(self, size):
    if not self._bpy_selectable:
        return 'left'
    return super().get_pref_col(size)