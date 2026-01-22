import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def findpartkey(self):
    """Get the part key for the latest numbered container seen."""
    numbered = self.numbered(self)
    if numbered and numbered.partkey:
        return numbered.partkey
    return ''