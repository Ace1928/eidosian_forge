from __future__ import annotations
import decimal
import re
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any, Callable
def value_node(value: int) -> tuple[Literal['value'], tuple[int]]:
    return ('value', (value,))