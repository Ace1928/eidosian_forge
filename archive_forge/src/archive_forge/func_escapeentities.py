import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def escapeentities(self, line):
    """Escape all Unicode characters to HTML entities."""
    result = ''
    pos = TextPosition(line)
    while not pos.finished():
        if ord(pos.current()) > 128:
            codepoint = hex(ord(pos.current()))
            if codepoint == '0xd835':
                codepoint = hex(ord(next(pos)) + 63488)
            result += '&#' + codepoint[1:] + ';'
        else:
            result += pos.current()
        pos.skipcurrent()
    return result