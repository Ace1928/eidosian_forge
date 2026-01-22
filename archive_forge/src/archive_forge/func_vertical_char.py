from __future__ import annotations
import io
import re
from html.parser import HTMLParser
from typing import Any
@vertical_char.setter
def vertical_char(self, val) -> None:
    val = str(val)
    self._validate_option('vertical_char', val)
    self._vertical_char = val