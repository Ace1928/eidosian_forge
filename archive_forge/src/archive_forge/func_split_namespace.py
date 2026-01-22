from __future__ import annotations
from datetime import datetime
from . import util
import re
from . import css_types as ct
import unicodedata
import bs4  # type: ignore[import]
from typing import Iterator, Iterable, Any, Callable, Sequence, cast  # noqa: F401
@staticmethod
def split_namespace(el: bs4.Tag, attr_name: str) -> tuple[str | None, str | None]:
    """Return namespace and attribute name without the prefix."""
    return (getattr(attr_name, 'namespace', None), getattr(attr_name, 'name', None))