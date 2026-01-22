import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def isinordered(self, type):
    """Find out if the layout type corresponds to an (un)ordered part."""
    return self.deasterisk(type).lower() in self.orderedlayouts