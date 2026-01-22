from __future__ import annotations
import re
import textwrap
import traceback
import typing as t
from .util import (
def complete_file(self) -> None:
    """Complete processing of the current file, if any."""
    if not self.file:
        return
    self.files.append(self.file)