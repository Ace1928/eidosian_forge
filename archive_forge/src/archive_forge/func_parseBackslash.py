from __future__ import absolute_import, unicode_literals, division
import re
import sys
from commonmark import common
from commonmark.common import normalize_uri, unescape_string
from commonmark.node import Node
from commonmark.normalize_reference import normalize_reference
def parseBackslash(self, block):
    """
        Parse a backslash-escaped special character, adding either the
        escaped character, a hard line break (if the backslash is followed
        by a newline), or a literal backslash to the block's children.
        Assumes current character is a backslash.
        """
    subj = self.subject
    self.pos += 1
    try:
        subjchar = subj[self.pos]
    except IndexError:
        subjchar = None
    if self.peek() == '\n':
        self.pos += 1
        node = Node('linebreak', None)
        block.append_child(node)
    elif subjchar and re.search(reEscapable, subjchar):
        block.append_child(text(subjchar))
        self.pos += 1
    else:
        block.append_child(text('\\'))
    return True