from __future__ import annotations
from collections.abc import Callable, MutableMapping
import dataclasses as dc
from typing import Any, Literal
import warnings
from markdown_it._compat import DATACLASS_KWARGS
def convert_attrs(value: Any) -> Any:
    """Convert Token.attrs set as ``None`` or ``[[key, value], ...]`` to a dict.

    This improves compatibility with upstream markdown-it.
    """
    if not value:
        return {}
    if isinstance(value, list):
        return dict(value)
    return value