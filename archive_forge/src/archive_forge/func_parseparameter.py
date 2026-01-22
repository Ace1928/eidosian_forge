import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def parseparameter(self, pos):
    """Parse a parameter at the current position"""
    self.factory.clearskipped(pos)
    if pos.finished():
        return None
    parameter = self.factory.parseany(pos)
    self.add(parameter)
    return parameter