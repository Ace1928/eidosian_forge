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
def format_progress_line(self) -> str:
    show_percent = self.show_percent
    info_bits = []
    if self.length is not None and show_percent is None:
        show_percent = not self.show_pos
    if self.show_pos:
        info_bits.append(self.format_pos())
    if show_percent:
        info_bits.append(self.format_pct())
    if self.show_eta and self.eta_known and (not self.finished):
        info_bits.append(self.format_eta())
    if self.item_show_func is not None:
        item_info = self.item_show_func(self.current_item)
        if item_info is not None:
            info_bits.append(item_info)
    return (self.bar_template % {'label': self.label, 'bar': self.format_bar(), 'info': self.info_sep.join(info_bits)}).rstrip()