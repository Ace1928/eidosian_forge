import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def processparameter(self, value):
    """Do the full processing on a parameter."""
    if not value:
        return None
    if self.extractnumber(value) == '0':
        return None
    for ignored in StyleConfig.size['ignoredtexts']:
        if ignored in value:
            value = value.replace(ignored, '')
    return value