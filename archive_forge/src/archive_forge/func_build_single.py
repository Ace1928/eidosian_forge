from __future__ import annotations
from . import util
from typing import TYPE_CHECKING, Any, Collection, NamedTuple
import re
import xml.etree.ElementTree as etree
from html import entities
def build_single(self, m: re.Match[str], tag: str, idx: int) -> etree.Element:
    """Return single tag."""
    el1 = etree.Element(tag)
    text = m.group(2)
    self.parse_sub_patterns(text, el1, None, idx)
    return el1