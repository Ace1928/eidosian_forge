import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def getlevel(self, type):
    """Get the level that corresponds to a layout type."""
    if self.isunique(type):
        return 0
    if not self.isinordered(type):
        Trace.error('Unknown layout type ' + type)
        return 0
    type = self.deasterisk(type).lower()
    level = self.orderedlayouts.index(type) + 1
    return level - DocumentParameters.startinglevel