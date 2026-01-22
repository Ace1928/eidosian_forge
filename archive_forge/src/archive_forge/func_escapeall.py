import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def escapeall(self, lines):
    """Escape all lines in an array according to the output options."""
    result = []
    for line in lines:
        if Options.html:
            line = self.escape(line, EscapeConfig.html)
        if Options.iso885915:
            line = self.escape(line, EscapeConfig.iso885915)
            line = self.escapeentities(line)
        elif not Options.str:
            line = self.escape(line, EscapeConfig.nonunicode)
        result.append(line)
    return result