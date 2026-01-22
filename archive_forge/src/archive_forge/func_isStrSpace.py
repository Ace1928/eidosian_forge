from __future__ import annotations
import re
from typing import Match, TypeVar
from .entities import entities
def isStrSpace(ch: str | None) -> bool:
    """Check if character is a whitespace."""
    return ch in ('\t', ' ')