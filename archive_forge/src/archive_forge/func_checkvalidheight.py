import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def checkvalidheight(self, container):
    """Check if the height parameter is valid; otherwise erase it."""
    heightspecial = container.getparameter('height_special')
    if self.height and self.extractnumber(self.height) == '1' and (heightspecial == 'totalheight'):
        self.height = None