from __future__ import absolute_import, unicode_literals, division
import re
import sys
from commonmark import common
from commonmark.common import normalize_uri, unescape_string
from commonmark.node import Node
from commonmark.normalize_reference import normalize_reference
def spnl(self):
    """ Parse zero or more space characters, including at
        most one newline."""
    self.match(reSpnl)
    return True