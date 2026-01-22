import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def styleparameter(self, name):
    """Get the style for a single parameter."""
    value = getattr(self, name)
    if value:
        return name.replace('max', 'max-') + ': ' + value + '; '
    return ''