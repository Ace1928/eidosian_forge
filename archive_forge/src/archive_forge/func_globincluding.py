import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def globincluding(self, magicchar):
    """Glob a bit of text up to (including) the magic char."""
    glob = self.glob(lambda: self.current() != magicchar) + magicchar
    self.skip(magicchar)
    return glob