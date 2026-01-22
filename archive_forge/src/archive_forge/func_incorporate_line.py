from __future__ import absolute_import, unicode_literals
import re
from commonmark import common
from commonmark.common import unescape_string
from commonmark.inlines import InlineParser
from commonmark.node import Node
def incorporate_line(self, ln):
    """Analyze a line of text and update the document appropriately.

        We parse markdown text by calling this on each line of input,
        then finalizing the document.
        """
    all_matched = True
    container = self.doc
    self.oldtip = self.tip
    self.offset = 0
    self.column = 0
    self.blank = False
    self.partially_consumed_tab = False
    self.line_number += 1
    if re.search('\\u0000', ln) is not None:
        ln = re.sub('\\0', '�', ln)
    self.current_line = ln
    while True:
        last_child = container.last_child
        if not (last_child and last_child.is_open):
            break
        container = last_child
        self.find_next_nonspace()
        rv = self.blocks[container.t].continue_(self, container)
        if rv == 0:
            pass
        elif rv == 1:
            all_matched = False
        elif rv == 2:
            self.last_line_length = len(ln)
            return
        else:
            raise ValueError('continue_ returned illegal value, must be 0, 1, or 2')
        if not all_matched:
            container = container.parent
            break
    self.all_closed = container == self.oldtip
    self.last_matched_container = container
    matched_leaf = container.t != 'paragraph' and self.blocks[container.t].accepts_lines
    starts = self.block_starts
    starts_len = len(starts.METHODS)
    while not matched_leaf:
        self.find_next_nonspace()
        if not self.indented and (not re.search(reMaybeSpecial, ln[self.next_nonspace:])):
            self.advance_next_nonspace()
            break
        i = 0
        while i < starts_len:
            res = getattr(starts, starts.METHODS[i])(self, container)
            if res == 1:
                container = self.tip
                break
            elif res == 2:
                container = self.tip
                matched_leaf = True
                break
            else:
                i += 1
        if i == starts_len:
            self.advance_next_nonspace()
            break
    if not self.all_closed and (not self.blank) and (self.tip.t == 'paragraph'):
        self.add_line()
    else:
        self.close_unmatched_blocks()
        if self.blank and container.last_child:
            container.last_child.last_line_blank = True
        t = container.t
        last_line_blank = self.blank and (not (t == 'block_quote' or (t == 'code_block' and container.is_fenced) or (t == 'item' and (not container.first_child) and (container.sourcepos[0][0] == self.line_number))))
        cont = container
        while cont:
            cont.last_line_blank = last_line_blank
            cont = cont.parent
        if self.blocks[t].accepts_lines:
            self.add_line()
            if t == 'html_block' and container.html_block_type >= 1 and (container.html_block_type <= 5) and re.search(reHtmlBlockClose[container.html_block_type], self.current_line[self.offset:]):
                self.finalize(container, self.line_number)
        elif self.offset < len(ln) and (not self.blank):
            container = self.add_child('paragraph', self.offset)
            self.advance_next_nonspace()
            self.add_line()
    self.last_line_length = len(ln)