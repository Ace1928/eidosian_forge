from __future__ import absolute_import, unicode_literals, division
import re
import sys
from commonmark import common
from commonmark.common import normalize_uri, unescape_string
from commonmark.node import Node
from commonmark.normalize_reference import normalize_reference
def parseEntity(self, block):
    """Attempt to parse an entity."""
    m = self.match(reEntityHere)
    if m:
        block.append_child(text(HTMLunescape(m)))
        return True
    else:
        return False