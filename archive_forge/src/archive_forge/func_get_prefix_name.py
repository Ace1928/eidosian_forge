from __future__ import annotations
from datetime import datetime
from . import util
import re
from . import css_types as ct
import unicodedata
import bs4  # type: ignore[import]
from typing import Iterator, Iterable, Any, Callable, Sequence, cast  # noqa: F401
@staticmethod
def get_prefix_name(el: bs4.Tag) -> str | None:
    """Get prefix."""
    return cast('str | None', el.prefix)