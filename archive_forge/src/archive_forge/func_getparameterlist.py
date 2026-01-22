import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def getparameterlist(self, name):
    """Get the value of a comma-separated parameter as a list."""
    paramtext = self.getparameter(name)
    if not paramtext:
        return []
    return paramtext.split(',')