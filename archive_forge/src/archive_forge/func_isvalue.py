import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def isvalue(self):
    """Return if the current character is a value character:"""
    'not a bracket or a space.'
    if self.current().isspace():
        return False
    if self.current() in '{}()':
        return False
    return True