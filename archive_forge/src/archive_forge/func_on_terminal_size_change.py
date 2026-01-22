from typing import (
from types import TracebackType
import logging
import re
import sys
import blessed
from .formatstring import fmtstr, FmtStr
from .formatstringarray import FSArray
from .termhelpers import Cbreak
def on_terminal_size_change(self, height: int, width: int) -> None:
    self._last_lines_by_row = {}
    self._last_rendered_width = width
    self._last_rendered_height = height