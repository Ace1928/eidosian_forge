from __future__ import absolute_import, unicode_literals, division
import re
import sys
from commonmark import common
from commonmark.common import normalize_uri, unescape_string
from commonmark.node import Node
from commonmark.normalize_reference import normalize_reference
def parseInlines(self, block):
    """
        Parse string content in block into inline children,
        using refmap to resolve references.
        """
    self.subject = block.string_content.strip()
    self.pos = 0
    self.delimiters = None
    self.brackets = None
    while self.parseInline(block):
        pass
    block.string_content = None
    self.processEmphasis(None)