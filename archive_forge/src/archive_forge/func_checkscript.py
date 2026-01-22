import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def checkscript(self, contents, index):
    """Check if the current element is a sub- or superscript."""
    return self.checkcommand(contents, index, SymbolFunction)