import typing as t
from contextlib import contextmanager
from gettext import gettext as _
from ._compat import term_len
from .parser import split_opt
def write_paragraph(self) -> None:
    """Writes a paragraph into the buffer."""
    if self.buffer:
        self.write('\n')