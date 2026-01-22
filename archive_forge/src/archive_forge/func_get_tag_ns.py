from __future__ import annotations
from datetime import datetime
from . import util
import re
from . import css_types as ct
import unicodedata
import bs4  # type: ignore[import]
from typing import Iterator, Iterable, Any, Callable, Sequence, cast  # noqa: F401
def get_tag_ns(self, el: bs4.Tag) -> str:
    """Get tag namespace."""
    if self.supports_namespaces():
        namespace = ''
        ns = self.get_uri(el)
        if ns:
            namespace = ns
    else:
        namespace = NS_XHTML
    return namespace