from __future__ import annotations
from . import util
from typing import TYPE_CHECKING, Any, Collection, NamedTuple
import re
import xml.etree.ElementTree as etree
from html import entities
def build_double(self, m: re.Match[str], tags: str, idx: int) -> etree.Element:
    """Return double tag."""
    tag1, tag2 = tags.split(',')
    el1 = etree.Element(tag1)
    el2 = etree.Element(tag2)
    text = m.group(2)
    self.parse_sub_patterns(text, el2, None, idx)
    el1.append(el2)
    if len(m.groups()) == 3:
        text = m.group(3)
        self.parse_sub_patterns(text, el1, el2, idx)
    return el1