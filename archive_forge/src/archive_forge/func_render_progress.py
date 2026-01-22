import time of Click down, some infrequently used functionality is
import contextlib
import math
import os
import sys
import time
import typing as t
from gettext import gettext as _
from io import StringIO
from types import TracebackType
from ._compat import _default_text_stdout
from ._compat import CYGWIN
from ._compat import get_best_encoding
from ._compat import isatty
from ._compat import open_stream
from ._compat import strip_ansi
from ._compat import term_len
from ._compat import WIN
from .exceptions import ClickException
from .utils import echo
def render_progress(self) -> None:
    import shutil
    if self.is_hidden:
        if self._last_line != self.label:
            self._last_line = self.label
            echo(self.label, file=self.file, color=self.color)
        return
    buf = []
    if self.autowidth:
        old_width = self.width
        self.width = 0
        clutter_length = term_len(self.format_progress_line())
        new_width = max(0, shutil.get_terminal_size().columns - clutter_length)
        if new_width < old_width:
            buf.append(BEFORE_BAR)
            buf.append(' ' * self.max_width)
            self.max_width = new_width
        self.width = new_width
    clear_width = self.width
    if self.max_width is not None:
        clear_width = self.max_width
    buf.append(BEFORE_BAR)
    line = self.format_progress_line()
    line_len = term_len(line)
    if self.max_width is None or self.max_width < line_len:
        self.max_width = line_len
    buf.append(line)
    buf.append(' ' * (clear_width - line_len))
    line = ''.join(buf)
    if line != self._last_line:
        self._last_line = line
        echo(line, file=self.file, color=self.color, nl=False)
        self.file.flush()