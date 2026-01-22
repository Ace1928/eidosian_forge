import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def parseliteral(self, pos):
    """Parse a literal bracket."""
    self.factory.clearskipped(pos)
    if not self.factory.detecttype(Bracket, pos):
        if not pos.isvalue():
            Trace.error('No literal parameter found at: ' + pos.identifier())
            return None
        return pos.globvalue()
    bracket = Bracket().setfactory(self.factory)
    self.add(bracket.parseliteral(pos))
    return bracket.literal