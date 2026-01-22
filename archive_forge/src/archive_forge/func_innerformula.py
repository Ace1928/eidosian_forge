import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def innerformula(self, pos):
    """Parse a whole formula inside the bracket"""
    while not pos.finished():
        self.add(self.factory.parseany(pos))