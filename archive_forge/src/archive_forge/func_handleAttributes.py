from __future__ import annotations
import re
from markdown.treeprocessors import Treeprocessor, isString
from markdown.extensions import Extension
from typing import TYPE_CHECKING
def handleAttributes(self, el: etree.Element, txt: str) -> str:
    """ Set attributes and return text without definitions. """

    def attributeCallback(match: re.Match[str]):
        el.set(match.group(1), match.group(2).replace('\n', ' '))
    return ATTR_RE.sub(attributeCallback, txt)