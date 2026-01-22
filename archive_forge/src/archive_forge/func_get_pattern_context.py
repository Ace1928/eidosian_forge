from __future__ import annotations
from functools import wraps, lru_cache
import warnings
import re
from typing import Callable, Any
def get_pattern_context(pattern: str, index: int) -> tuple[str, int, int]:
    """Get the pattern context."""
    last = 0
    current_line = 1
    col = 1
    text = []
    line = 1
    offset = None
    for m in RE_PATTERN_LINE_SPLIT.finditer(pattern):
        linetext = pattern[last:m.start(0)]
        if not len(m.group(0)) and (not len(text)):
            indent = ''
            offset = -1
            col = index - last + 1
        elif last <= index < m.end(0):
            indent = '--> '
            offset = (-1 if index > m.start(0) else 0) + 3
            col = index - last + 1
        else:
            indent = '    '
            offset = None
        if len(text):
            text.append('\n')
        text.append(f'{indent}{linetext}')
        if offset is not None:
            text.append('\n')
            text.append(' ' * (col + offset) + '^')
            line = current_line
        current_line += 1
        last = m.end(0)
    return (''.join(text), line, col)