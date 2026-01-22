from __future__ import absolute_import, unicode_literals
import re
from commonmark import common
from commonmark.common import unescape_string
from commonmark.inlines import InlineParser
from commonmark.node import Node
def ends_with_blank_line(block):
    """ Returns true if block ends with a blank line,
    descending if needed into lists and sublists."""
    while block:
        if block.last_line_blank:
            return True
        if not block.last_line_checked and block.t in ('list', 'item'):
            block.last_line_checked = True
            block = block.last_child
        else:
            block.last_line_checked = True
            break
    return False