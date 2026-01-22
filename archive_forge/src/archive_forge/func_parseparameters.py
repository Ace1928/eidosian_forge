import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def parseparameters(self, pos, macro):
    """Parse as many parameters as are needed."""
    self.parseoptional(pos, list(macro.defaults))
    self.parsemandatory(pos, macro.parameternumber - len(macro.defaults))
    if len(self.values) < macro.parameternumber:
        Trace.error('Missing parameters in macro ' + str(self))