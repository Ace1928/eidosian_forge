from __future__ import annotations
from collections.abc import Callable, MutableMapping
import dataclasses as dc
from typing import Any, Literal
import warnings
from markdown_it._compat import DATACLASS_KWARGS
def attrGet(self, name: str) -> None | str | int | float:
    """Get the value of attribute `name`, or null if it does not exist."""
    return self.attrs.get(name, None)