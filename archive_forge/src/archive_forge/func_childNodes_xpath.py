from __future__ import annotations
from copy import copy, deepcopy
from xml.dom import Node
from xml.dom.minidom import Attr, NamedNodeMap
from lxml.etree import (
from lxml.html import HtmlElementClassLookup, HTMLParser
@property
def childNodes_xpath(self):
    for n in self._xp_childrennodes(self):
        if isinstance(n, ElementBase):
            yield n
        elif isinstance(n, (_ElementStringResult, _ElementUnicodeResult)):
            if isinstance(n, _ElementUnicodeResult):
                n = DomElementUnicodeResult(n)
            else:
                n.nodeType = Node.TEXT_NODE
                n.data = n
            yield n