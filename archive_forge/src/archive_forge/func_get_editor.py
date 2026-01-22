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
def get_editor(self) -> str:
    if self.editor is not None:
        return self.editor
    for key in ('VISUAL', 'EDITOR'):
        rv = os.environ.get(key)
        if rv:
            return rv
    if WIN:
        return 'notepad'
    for editor in ('sensible-editor', 'vim', 'nano'):
        if os.system(f'which {editor} >/dev/null 2>&1') == 0:
            return editor
    return 'vi'