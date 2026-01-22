import re
from dataclasses import dataclass
from itertools import chain
from typing import Optional, Tuple
from .lazyre import LazyReCompile
def current_object(cursor_offset: int, line: str) -> Optional[LinePart]:
    """If in attribute completion, the object on which attribute should be
    looked up."""
    match = current_word(cursor_offset, line)
    if match is None:
        return None
    s = '.'.join((m.group(1) for m in _current_object_re.finditer(match.word) if m.end(1) + match.start < cursor_offset))
    if not s:
        return None
    return LinePart(match.start, match.start + len(s), s)