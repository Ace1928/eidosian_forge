from __future__ import annotations
import io
import re
from html.parser import HTMLParser
from typing import Any
@property
def dividers(self) -> list[bool]:
    return self._dividers[:]