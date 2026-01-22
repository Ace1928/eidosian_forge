import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def writefunction(self, pos):
    """Write a single function f0,...,fn."""
    tag = self.readtag(pos)
    if not tag:
        return None
    if pos.checkskip('/'):
        return TaggedBit().selfcomplete(tag)
    if not pos.checkskip('{'):
        Trace.error('Function should be defined in {}')
        return None
    pos.pushending('}')
    contents = self.writepos(pos)
    pos.popending()
    if len(contents) == 0:
        return None
    return TaggedBit().complete(contents, tag)