from __future__ import annotations
import re
from ..common.utils import charStrAt, isStrSpace
from .state_block import StateBlock
def escapedSplit(string: str) -> list[str]:
    result: list[str] = []
    pos = 0
    max = len(string)
    isEscaped = False
    lastPos = 0
    current = ''
    ch = charStrAt(string, pos)
    while pos < max:
        if ch == '|':
            if not isEscaped:
                result.append(current + string[lastPos:pos])
                current = ''
                lastPos = pos + 1
            else:
                current += string[lastPos:pos - 1]
                lastPos = pos
        isEscaped = ch == '\\'
        pos += 1
        ch = charStrAt(string, pos)
    result.append(current + string[lastPos:])
    return result