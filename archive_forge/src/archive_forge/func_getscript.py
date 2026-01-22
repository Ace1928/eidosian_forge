import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def getscript(self, contents, index):
    """Get the sub- or superscript."""
    bit = contents[index]
    bit.output.tag += ' class="script"'
    del contents[index]
    return bit