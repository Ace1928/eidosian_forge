from __future__ import annotations
from collections.abc import Callable, MutableMapping
import dataclasses as dc
from typing import Any, Literal
import warnings
from markdown_it._compat import DATACLASS_KWARGS
def attrItems(self) -> list[tuple[str, str | int | float]]:
    """Get (key, value) list of attrs."""
    return list(self.attrs.items())