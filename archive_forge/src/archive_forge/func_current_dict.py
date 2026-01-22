import re
from dataclasses import dataclass
from itertools import chain
from typing import Optional, Tuple
from .lazyre import LazyReCompile
def current_dict(cursor_offset: int, line: str) -> Optional[LinePart]:
    """If in dictionary completion, return the dict that should be used"""
    for m in _current_dict_re.finditer(line):
        if m.start(2) <= cursor_offset <= m.end(2):
            return LinePart(m.start(1), m.end(1), m.group(1))
    return None