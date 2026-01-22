from __future__ import annotations
import re
import textwrap
import traceback
import typing as t
from .util import (
def process_continue(self) -> None:
    """Process a diff start, range or header line."""
    if self.line.startswith('diff '):
        self.process_start()
    elif self.line.startswith('@@ '):
        self.process_range()
    else:
        self.process_header()