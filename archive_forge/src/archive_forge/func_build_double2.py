from __future__ import annotations
from . import util
from typing import TYPE_CHECKING, Any, Collection, NamedTuple
import re
import xml.etree.ElementTree as etree
from html import entities
def build_double2(self, m: re.Match[str], tags: str, idx: int) -> etree.Element:
    """Return double tags (variant 2): `<strong>text <em>text</em></strong>`."""
    tag1, tag2 = tags.split(',')
    el1 = etree.Element(tag1)
    el2 = etree.Element(tag2)
    text = m.group(2)
    self.parse_sub_patterns(text, el1, None, idx)
    text = m.group(3)
    el1.append(el2)
    self.parse_sub_patterns(text, el2, None, idx)
    return el1