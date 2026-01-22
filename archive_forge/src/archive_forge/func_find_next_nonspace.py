from __future__ import absolute_import, unicode_literals
import re
from commonmark import common
from commonmark.common import unescape_string
from commonmark.inlines import InlineParser
from commonmark.node import Node
def find_next_nonspace(self):
    current_line = self.current_line
    i = self.offset
    cols = self.column
    try:
        c = current_line[i]
    except IndexError:
        c = ''
    while c != '':
        if c == ' ':
            i += 1
            cols += 1
        elif c == '\t':
            i += 1
            cols += 4 - cols % 4
        else:
            break
        try:
            c = current_line[i]
        except IndexError:
            c = ''
    self.blank = c == '\n' or c == '\r' or c == ''
    self.next_nonspace = i
    self.next_nonspace_column = cols
    self.indent = self.next_nonspace_column - self.column
    self.indented = self.indent >= CODE_INDENT