from __future__ import annotations
from datetime import datetime
from . import util
import re
from . import css_types as ct
import unicodedata
import bs4  # type: ignore[import]
from typing import Iterator, Iterable, Any, Callable, Sequence, cast  # noqa: F401
@classmethod
def is_content_string(cls, obj: bs4.PageElement) -> bool:
    """Check if node is content string."""
    return cls.is_navigable_string(obj) and (not cls.is_special_string(obj))