from __future__ import annotations
from datetime import datetime
from . import util
import re
from . import css_types as ct
import unicodedata
import bs4  # type: ignore[import]
from typing import Iterator, Iterable, Any, Callable, Sequence, cast  # noqa: F401
def is_iframe(self, el: bs4.Tag) -> bool:
    """Check if element is an `iframe`."""
    return bool((el.name if self.is_xml_tree(el) else util.lower(el.name)) == 'iframe' and self.is_html_tag(el))