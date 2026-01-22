from __future__ import annotations
from datetime import datetime
from . import util
import re
from . import css_types as ct
import unicodedata
import bs4  # type: ignore[import]
from typing import Iterator, Iterable, Any, Callable, Sequence, cast  # noqa: F401
def get_own_text(self, el: bs4.Tag, no_iframe: bool=False) -> list[str]:
    """Get Own Text."""
    return [node for node in self.get_contents(el, no_iframe=no_iframe) if self.is_content_string(node)]