from __future__ import annotations
from datetime import datetime
from . import util
import re
from . import css_types as ct
import unicodedata
import bs4  # type: ignore[import]
from typing import Iterator, Iterable, Any, Callable, Sequence, cast  # noqa: F401
def match_placeholder_shown(self, el: bs4.Tag) -> bool:
    """
        Match placeholder shown according to HTML spec.

        - text area should be checked if they have content. A single newline does not count as content.

        """
    match = False
    content = self.get_text(el)
    if content in ('', '\n'):
        match = True
    return match