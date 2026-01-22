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
def format_eta(self) -> str:
    if self.eta_known:
        t = int(self.eta)
        seconds = t % 60
        t //= 60
        minutes = t % 60
        t //= 60
        hours = t % 24
        t //= 24
        if t > 0:
            return f'{t}d {hours:02}:{minutes:02}:{seconds:02}'
        else:
            return f'{hours:02}:{minutes:02}:{seconds:02}'
    return ''