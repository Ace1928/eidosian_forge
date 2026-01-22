from __future__ import annotations
import io
import re
from html.parser import HTMLParser
from typing import Any
@vrules.setter
def vrules(self, val) -> None:
    self._validate_option('vrules', val)
    self._vrules = val