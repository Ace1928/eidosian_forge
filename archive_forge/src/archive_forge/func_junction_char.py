from __future__ import annotations
import io
import re
from html.parser import HTMLParser
from typing import Any
@junction_char.setter
def junction_char(self, val) -> None:
    val = str(val)
    self._validate_option('junction_char', val)
    self._junction_char = val