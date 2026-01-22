import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def getparameter(self, name):
    """Get the value of a parameter, if present."""
    if not name in self.parameters:
        return None
    return self.parameters[name]