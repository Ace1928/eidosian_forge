import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def parsecommandtype(self, command, type, pos):
    """Parse a given command type."""
    bit = self.factory.create(type)
    bit.setcommand(command)
    returned = bit.parsebit(pos)
    if returned:
        return returned
    return bit