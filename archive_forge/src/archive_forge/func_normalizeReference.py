from __future__ import annotations
import re
from typing import Match, TypeVar
from .entities import entities
def normalizeReference(string: str) -> str:
    """Helper to unify [reference labels]."""
    string = re.sub('\\s+', ' ', string.strip())
    return string.lower().upper()