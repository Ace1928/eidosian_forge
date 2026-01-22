import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def removepercentwidth(self):
    """Remove percent width if present, to set it at the figure level."""
    if not self.width:
        return None
    if not '%' in self.width:
        return None
    width = self.width
    self.width = None
    if self.height == 'auto':
        self.height = None
    return width