from __future__ import annotations
from enum import Enum
import re
from typing import Callable
def handle_scanning_value(char: str, pos: int, tokens: TokenState) -> State:
    if char == '"':
        tokens.set_start(pos)
        return State.SCANNING_QUOTED_VALUE
    if REGEX_KEY_CHARACTERS.fullmatch(char):
        tokens.set_start(pos)
        return State.SCANNING_BARE_VALUE
    raise ParseError(f'Unexpected character whilst scanning value: {char}', pos)