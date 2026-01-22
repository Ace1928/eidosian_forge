from __future__ import absolute_import, unicode_literals
import re
from commonmark import common
from commonmark.common import unescape_string
from commonmark.inlines import InlineParser
from commonmark.node import Node
@staticmethod
def indented_code_block(parser, container=None):
    if parser.indented and parser.tip.t != 'paragraph' and (not parser.blank):
        parser.advance_offset(CODE_INDENT, True)
        parser.close_unmatched_blocks()
        parser.add_child('code_block', parser.offset)
        return 2
    return 0