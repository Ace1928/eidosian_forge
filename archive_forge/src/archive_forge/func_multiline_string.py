from __future__ import annotations
import argparse
import ast
import functools
import logging
import tokenize
from typing import Any
from typing import Generator
from typing import List
from typing import Tuple
from flake8 import defaults
from flake8 import utils
from flake8._compat import FSTRING_END
from flake8._compat import FSTRING_MIDDLE
from flake8.plugins.finder import LoadedPlugin
def multiline_string(self, token: tokenize.TokenInfo) -> Generator[str, None, None]:
    """Iterate through the lines of a multiline string."""
    if token.type == FSTRING_END:
        start = self._fstring_start
    else:
        start = token.start[0]
    self.multiline = True
    self.line_number = start
    for _ in range(start, token.end[0]):
        yield self.lines[self.line_number - 1]
        self.line_number += 1
    self.multiline = False