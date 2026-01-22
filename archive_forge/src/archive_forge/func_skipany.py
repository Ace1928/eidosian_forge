import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def skipany(self, pos):
    """Skip any skipped types."""
    for type in self.skippedtypes:
        if self.instance(type).detect(pos):
            return self.parsetype(type, pos)
    return None