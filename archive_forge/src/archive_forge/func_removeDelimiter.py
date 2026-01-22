from __future__ import absolute_import, unicode_literals, division
import re
import sys
from commonmark import common
from commonmark.common import normalize_uri, unescape_string
from commonmark.node import Node
from commonmark.normalize_reference import normalize_reference
def removeDelimiter(self, delim):
    if delim.get('previous') is not None:
        delim['previous']['next'] = delim.get('next')
    if delim.get('next') is None:
        self.delimiters = delim.get('previous')
    else:
        delim['next']['previous'] = delim.get('previous')