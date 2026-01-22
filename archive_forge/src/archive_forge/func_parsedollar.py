import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def parsedollar(self, pos):
    """Parse to the next $."""
    pos.pushending('$')
    self.parsed = pos.globexcluding('$')
    pos.popending('$')