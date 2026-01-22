from __future__ import absolute_import, unicode_literals, division
import re
import sys
from commonmark import common
from commonmark.common import normalize_uri, unescape_string
from commonmark.node import Node
from commonmark.normalize_reference import normalize_reference
def parseBang(self, block):
    """
        If next character is [, and ! delimiter to delimiter stack and
        add a text node to block's children. Otherwise just add a text
        node.
        """
    startpos = self.pos
    self.pos += 1
    if self.peek() == '[':
        self.pos += 1
        node = text('![')
        block.append_child(node)
        self.addBracket(node, startpos + 1, True)
    else:
        block.append_child(text('!'))
    return True