from __future__ import annotations
from datetime import datetime
from . import util
import re
from . import css_types as ct
import unicodedata
import bs4  # type: ignore[import]
from typing import Iterator, Iterable, Any, Callable, Sequence, cast  # noqa: F401
def match_root(self, el: bs4.Tag) -> bool:
    """Match element as root."""
    is_root = self.is_root(el)
    if is_root:
        sibling = self.get_previous(el, tags=False)
        while is_root and sibling is not None:
            if self.is_tag(sibling) or (self.is_content_string(sibling) and sibling.strip()) or self.is_cdata(sibling):
                is_root = False
            else:
                sibling = self.get_previous(sibling, tags=False)
    if is_root:
        sibling = self.get_next(el, tags=False)
        while is_root and sibling is not None:
            if self.is_tag(sibling) or (self.is_content_string(sibling) and sibling.strip()) or self.is_cdata(sibling):
                is_root = False
            else:
                sibling = self.get_next(sibling, tags=False)
    return is_root