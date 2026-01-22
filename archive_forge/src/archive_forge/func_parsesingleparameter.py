import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def parsesingleparameter(self, pos):
    """Parse a parameter, or a single letter."""
    self.factory.clearskipped(pos)
    if pos.finished():
        Trace.error('Error while parsing single parameter at ' + pos.identifier())
        return None
    if self.factory.detecttype(Bracket, pos) or self.factory.detecttype(FormulaCommand, pos):
        return self.parseparameter(pos)
    letter = FormulaConstant(pos.skipcurrent())
    self.add(letter)
    return letter