import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def setmaster(self, master):
    """Set the master counter."""
    self.master = master
    self.last = self.master.getvalue()
    return self