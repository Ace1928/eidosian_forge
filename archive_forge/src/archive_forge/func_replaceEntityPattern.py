from __future__ import annotations
import re
from typing import Match, TypeVar
from .entities import entities
def replaceEntityPattern(match: str, name: str) -> str:
    """Convert HTML entity patterns,
    see https://spec.commonmark.org/0.30/#entity-references
    """
    if name in entities:
        return entities[name]
    code: None | int = None
    if (pat := DIGITAL_ENTITY_BASE10_RE.fullmatch(name)):
        code = int(pat.group(1), 10)
    elif (pat := DIGITAL_ENTITY_BASE16_RE.fullmatch(name)):
        code = int(pat.group(1), 16)
    if code is not None and isValidEntityCode(code):
        return fromCodePoint(code)
    return match