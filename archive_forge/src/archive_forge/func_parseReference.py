from __future__ import absolute_import, unicode_literals, division
import re
import sys
from commonmark import common
from commonmark.common import normalize_uri, unescape_string
from commonmark.node import Node
from commonmark.normalize_reference import normalize_reference
def parseReference(self, s, refmap):
    """Attempt to parse a link reference, modifying refmap."""
    self.subject = s
    self.pos = 0
    startpos = self.pos
    match_chars = self.parseLinkLabel()
    if match_chars == 0 or match_chars == 2:
        return 0
    else:
        rawlabel = self.subject[:match_chars]
    if self.peek() == ':':
        self.pos += 1
    else:
        self.pos = startpos
        return 0
    self.spnl()
    dest = self.parseLinkDestination()
    if dest is None:
        self.pos = startpos
        return 0
    beforetitle = self.pos
    self.spnl()
    title = None
    if self.pos != beforetitle:
        title = self.parseLinkTitle()
    if title is None:
        title = ''
        self.pos = beforetitle
    at_line_end = True
    if self.match(reSpaceAtEndOfLine) is None:
        if title == '':
            at_line_end = False
        else:
            title == ''
            self.pos = beforetitle
            at_line_end = self.match(reSpaceAtEndOfLine) is not None
    if not at_line_end:
        self.pos = startpos
        return 0
    normlabel = normalize_reference(rawlabel)
    if normlabel == '':
        self.pos = startpos
        return 0
    if not refmap.get(normlabel):
        refmap[normlabel] = {'destination': dest, 'title': title}
    return self.pos - startpos