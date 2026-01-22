from __future__ import annotations
from datetime import datetime
from . import util
import re
from . import css_types as ct
import unicodedata
import bs4  # type: ignore[import]
from typing import Iterator, Iterable, Any, Callable, Sequence, cast  # noqa: F401
def match_attributes(self, el: bs4.Tag, attributes: tuple[ct.SelectorAttribute, ...]) -> bool:
    """Match attributes."""
    match = True
    if attributes:
        for a in attributes:
            temp = self.match_attribute_name(el, a.attribute, a.prefix)
            pattern = a.xml_type_pattern if self.is_xml and a.xml_type_pattern else a.pattern
            if temp is None:
                match = False
                break
            value = temp if isinstance(temp, str) else ' '.join(temp)
            if pattern is None:
                continue
            elif pattern.match(value) is None:
                match = False
                break
    return match