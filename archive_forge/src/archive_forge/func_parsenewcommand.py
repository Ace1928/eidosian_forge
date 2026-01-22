import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def parsenewcommand(self, pos):
    """Parse the name of the new command."""
    self.factory.clearskipped(pos)
    if self.factory.detecttype(Bracket, pos):
        return self.parseliteral(pos)
    if self.factory.detecttype(FormulaCommand, pos):
        return self.factory.create(FormulaCommand).extractcommand(pos)
    Trace.error('Unknown formula bit in defining function at ' + pos.identifier())
    return 'unknown'