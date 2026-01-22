from __future__ import annotations
from collections.abc import Generator, Sequence
import textwrap
from typing import Any, NamedTuple, TypeVar, overload
from .token import Token
@property
def previous_sibling(self: _NodeType) -> _NodeType | None:
    """Get the previous node in the sequence of siblings.

        Returns `None` if this is the first sibling.
        """
    self_index = self.siblings.index(self)
    if self_index - 1 >= 0:
        return self.siblings[self_index - 1]
    return None