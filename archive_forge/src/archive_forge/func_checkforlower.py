import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def checkforlower(self, string):
    """Check for a string in lower case."""
    extracted = self.extract(len(string))
    if not extracted:
        return False
    return string.lower() == self.extract(len(string)).lower()