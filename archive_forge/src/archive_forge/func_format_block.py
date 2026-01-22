from __future__ import annotations
import collections.abc as c
import datetime
import typing as t
from .util import (
from .util_common import (
from .metadata import (
from .config import (
from . import junit_xml
def format_block(self) -> str:
    """Format the test summary or messages as a block of text and return the result."""
    if self.summary:
        block = self.summary
    else:
        block = '\n'.join((m.format() for m in self.messages))
    message = block.strip()
    message = message.replace(display.clear, '')
    return message