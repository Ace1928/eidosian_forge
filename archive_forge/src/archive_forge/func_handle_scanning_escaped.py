from __future__ import annotations
from enum import Enum
import re
from typing import Callable
def handle_scanning_escaped(char: str, pos: int, tokens: TokenState) -> State:
    return State.SCANNING_QUOTED_VALUE