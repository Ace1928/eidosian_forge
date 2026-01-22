import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def checkdirection(self, bit, command):
    """Check if the given bit is the desired bracket command."""
    if not isinstance(bit, BracketCommand):
        return False
    return bit.command == command