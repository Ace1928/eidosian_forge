from __future__ import annotations
import decimal
import re
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any, Callable
def skip_token(tokens: list[tuple[str, str]], type_: str, value: str | None=None):
    if test_next_token(tokens, type_, value):
        return tokens.pop()