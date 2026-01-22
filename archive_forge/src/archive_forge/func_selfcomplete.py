import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def selfcomplete(self, tag):
    """Set the self-closing tag, no contents (as in <hr/>)."""
    self.output = TaggedOutput().settag(tag, empty=True)
    return self