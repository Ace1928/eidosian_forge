from __future__ import annotations
import re
from collections.abc import Generator
from typing import NamedTuple
def indicates_division(token: Token) -> bool:
    """A helper function that helps the tokenizer to decide if the current
    token may be followed by a division operator.
    """
    if token.type == 'operator':
        return token.value in (')', ']', '}', '++', '--')
    return token.type in ('name', 'number', 'string', 'regexp')