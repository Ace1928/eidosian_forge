import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def getliteralvalue(self, name):
    """Get the literal value of a parameter."""
    param = self.getparam(name)
    if not param or not param.literalvalue:
        return None
    return param.literalvalue