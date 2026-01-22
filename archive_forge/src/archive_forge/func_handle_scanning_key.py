from __future__ import annotations
from enum import Enum
import re
from typing import Callable
def handle_scanning_key(char: str, pos: int, tokens: TokenState) -> State:
    if char == '=':
        tokens.append(tokens.start, pos, 'key')
        return State.SCANNING_VALUE
    if REGEX_KEY_CHARACTERS.fullmatch(char):
        return State.SCANNING_KEY
    raise ParseError(f'Unexpected character whilst scanning key: {char}', pos)