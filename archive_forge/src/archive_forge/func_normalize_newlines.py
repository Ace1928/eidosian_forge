from __future__ import annotations
import re
from typing import TYPE_CHECKING, Type
from attrs import define
from ufoLib2.serde import serde
def normalize_newlines(self) -> Features:
    """Normalize CRLF and CR newlines to just LF."""
    self.text = RE_NEWLINES.sub('\n', self.text)
    return self