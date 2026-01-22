import contextlib
import re
from dataclasses import dataclass
from typing import Dict, Iterator, NoReturn, Optional, Tuple, Union
from .specifiers import Specifier
@contextlib.contextmanager
def enclosing_tokens(self, open_token: str, close_token: str, *, around: str) -> Iterator[None]:
    if self.check(open_token):
        open_position = self.position
        self.read()
    else:
        open_position = None
    yield
    if open_position is None:
        return
    if not self.check(close_token):
        self.raise_syntax_error(f'Expected matching {close_token} for {open_token}, after {around}', span_start=open_position)
    self.read()