from __future__ import annotations
from . import Extension
from ..treeprocessors import Treeprocessor
from ..util import parseBoolValue, AMP_SUBSTITUTE, deprecated, HTML_PLACEHOLDER_RE, AtomicString
from ..treeprocessors import UnescapeTreeprocessor
from ..serializers import RE_AMP
import re
import html
import unicodedata
from copy import deepcopy
import xml.etree.ElementTree as etree
from typing import TYPE_CHECKING, Any, Iterator, MutableSet
def iterparent(self, node: etree.Element) -> Iterator[tuple[etree.Element, etree.Element]]:
    """ Iterator wrapper to get allowed parent and child all at once. """
    for child in node:
        if not self.header_rgx.match(child.tag) and child.tag not in ['pre', 'code']:
            yield (node, child)
            yield from self.iterparent(child)