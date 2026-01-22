import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def findright(self, contents, index):
    """Find the right bracket starting at the given index, or 0."""
    depth = 1
    while index < len(contents):
        if self.checkleft(contents, index):
            depth += 1
        if self.checkright(contents, index):
            depth -= 1
        if depth == 0:
            return index
        index += 1
    return None