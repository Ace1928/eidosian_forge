from __future__ import annotations
import decimal
import re
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any, Callable
def range_or_value(self):
    left = self.value()
    if skip_token(self.tokens, 'ellipsis'):
        return (left, self.value())
    else:
        return (left, left)