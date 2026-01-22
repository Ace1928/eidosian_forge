import re
from dataclasses import dataclass
from itertools import chain
from typing import Optional, Tuple
from .lazyre import LazyReCompile
def current_dotted_attribute(cursor_offset: int, line: str) -> Optional[LinePart]:
    """The dotted attribute-object pair before the cursor"""
    match = current_word(cursor_offset, line)
    if match is not None and '.' in match.word[1:]:
        return match
    return None