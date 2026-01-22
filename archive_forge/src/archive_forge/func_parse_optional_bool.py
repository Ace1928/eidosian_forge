from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import (
@staticmethod
def parse_optional_bool(v: str) -> Optional[bool]:
    s = str(v).lower()
    if s in {'0', 'no', 'false'}:
        return False
    if s in {'1', 'yes', 'true'}:
        return True
    if s in {'auto', 'none'}:
        return None
    raise ValueError('invalid optional bool: {v!r}')