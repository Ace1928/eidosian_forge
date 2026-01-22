import re
from dataclasses import dataclass
from itertools import chain
from typing import Optional, Tuple
from .lazyre import LazyReCompile
def current_method_definition_name(cursor_offset: int, line: str) -> Optional[LinePart]:
    """The name of a method being defined"""
    for m in _current_method_definition_name_re.finditer(line):
        if m.start(1) <= cursor_offset <= m.end(1):
            return LinePart(m.start(1), m.end(1), m.group(1))
    return None