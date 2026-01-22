import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def parsesquareliteral(self, pos):
    """Parse a square bracket literally."""
    self.factory.clearskipped(pos)
    if not self.factory.detecttype(SquareBracket, pos):
        return None
    bracket = SquareBracket().setfactory(self.factory)
    self.add(bracket.parseliteral(pos))
    return bracket.literal