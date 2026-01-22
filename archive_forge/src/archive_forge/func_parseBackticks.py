from __future__ import absolute_import, unicode_literals, division
import re
import sys
from commonmark import common
from commonmark.common import normalize_uri, unescape_string
from commonmark.node import Node
from commonmark.normalize_reference import normalize_reference
def parseBackticks(self, block):
    """ Attempt to parse backticks, adding either a backtick code span or a
        literal sequence of backticks to the 'inlines' list."""
    ticks = self.match(reTicksHere)
    if ticks is None:
        return False
    after_open_ticks = self.pos
    matched = self.match(reTicks)
    while matched is not None:
        if matched == ticks:
            node = Node('code', None)
            contents = self.subject[after_open_ticks:self.pos - len(ticks)].replace('\n', ' ')
            if contents.lstrip(' ') and contents[0] == contents[-1] == ' ':
                node.literal = contents[1:-1]
            else:
                node.literal = contents
            block.append_child(node)
            return True
        matched = self.match(reTicks)
    self.pos = after_open_ticks
    block.append_child(text(ticks))
    return True