from __future__ import annotations
from enum import Enum
import re
from typing import Callable
def handle_scanning_bare_value(char: str, pos: int, tokens: TokenState) -> State:
    if REGEX_KEY_CHARACTERS.fullmatch(char):
        return State.SCANNING_BARE_VALUE
    if char == '}':
        tokens.append(tokens.start, pos, 'value')
        return State.DONE
    if REGEX_SPACE.fullmatch(char):
        tokens.append(tokens.start, pos, 'value')
        return State.SCANNING
    raise ParseError(f'Unexpected character whilst scanning bare value: {char}', pos)