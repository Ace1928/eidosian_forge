import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def modifylimits(self, contents, index):
    """Modify a limits commands so that the limits appear above and below."""
    limited = contents[index]
    subscript = self.getlimit(contents, index + 1)
    limited.contents.append(subscript)
    if self.checkscript(contents, index + 1):
        superscript = self.getlimit(contents, index + 1)
    else:
        superscript = TaggedBit().constant('\u205f', 'sup class="limit"')
    limited.contents.insert(0, superscript)