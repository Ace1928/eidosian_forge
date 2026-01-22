from __future__ import annotations
import io
import re
from html.parser import HTMLParser
from typing import Any
@left_padding_width.setter
def left_padding_width(self, val) -> None:
    self._validate_option('left_padding_width', val)
    self._left_padding_width = val