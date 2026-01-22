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
def on_edit_pos_changed(self, edit, position):
    """Gets called when the cursor position inside the edit changed.
        Rehighlight the current line because there might be a paren under
        the cursor now."""
    tokens = self.tokenize(self.current_line, False)
    edit.set_edit_markup(list(format_tokens(tokens)))