from __future__ import annotations
import re
from typing import Match, TypeVar
from .entities import entities
def isLinkOpen(string: str) -> bool:
    return bool(LINK_OPEN_RE.search(string))