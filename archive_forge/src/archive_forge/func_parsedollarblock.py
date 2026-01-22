import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def parsedollarblock(self, pos):
    """Parse a $$...$$ formula."""
    self.header = ['block']
    self.parsedollar(pos)
    if not pos.checkskip('$'):
        pos.error('Formula should be $$...$$, but last $ is missing.')