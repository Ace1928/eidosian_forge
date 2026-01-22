import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def nextending(self):
    """Return the next ending in the queue."""
    nextending = self.endinglist.findending(self)
    if not nextending:
        return None
    return nextending.ending