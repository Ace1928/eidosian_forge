import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def getmessage(self, key):
    """Get the translated message for the given key."""
    if self.first:
        self.findtranslation()
        self.first = False
    message = self.getuntranslated(key)
    if not self.translation:
        return message
    try:
        message = self.translation.ugettext(message)
    except IOError:
        pass
    return message