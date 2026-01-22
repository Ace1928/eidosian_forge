import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def readparams(self, readtemplate, pos):
    """Read the params according to the template."""
    self.params = dict()
    for paramdef in self.paramdefs(readtemplate):
        paramdef.read(pos, self)
        self.params['$' + paramdef.name] = paramdef