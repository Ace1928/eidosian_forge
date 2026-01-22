import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def getuntranslated(self, key):
    """Get the untranslated message."""
    if not key in TranslationConfig.constants:
        Trace.error('Cannot translate ' + key)
        return key
    return TranslationConfig.constants[key]