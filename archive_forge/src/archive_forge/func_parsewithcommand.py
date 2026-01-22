import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def parsewithcommand(self, command, pos):
    """Parse the command type once we have the command."""
    for type in FormulaCommand.types:
        if command in type.commandmap:
            return self.parsecommandtype(command, type, pos)
    return None