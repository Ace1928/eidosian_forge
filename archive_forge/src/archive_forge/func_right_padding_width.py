from __future__ import annotations
import io
import re
from html.parser import HTMLParser
from typing import Any
@right_padding_width.setter
def right_padding_width(self, val) -> None:
    self._validate_option('right_padding_width', val)
    self._right_padding_width = val