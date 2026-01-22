from __future__ import annotations
from . import util
from typing import TYPE_CHECKING, Any, Collection, NamedTuple
import re
import xml.etree.ElementTree as etree
from html import entities
def backslash_unescape(self, text: str) -> str:
    """ Return text with backslash escapes undone (backslashes are restored). """
    try:
        RE = self.md.treeprocessors['unescape'].RE
    except KeyError:
        return text

    def _unescape(m: re.Match[str]) -> str:
        return chr(int(m.group(1)))
    return RE.sub(_unescape, text)