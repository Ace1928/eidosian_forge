import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def formatcontents(self):
    """Format the reference contents."""
    formatkey = self.getparameter('LatexCommand')
    if not formatkey:
        formatkey = 'ref'
    self.formatted = '↕'
    if formatkey in StyleConfig.referenceformats:
        self.formatted = StyleConfig.referenceformats[formatkey]
    else:
        Trace.error('Unknown reference format ' + formatkey)
    self.replace('↕', self.direction)
    self.replace('#', '1')
    self.replace('on-page', Translator.translate('on-page'))
    partkey = self.destination.findpartkey()
    self.replace('@', partkey and partkey.number)
    self.replace('¶', partkey and partkey.tocentry)
    if not '$' in self.formatted or not partkey or (not partkey.titlecontents):
        self.contents = [Constant(self.formatted)]
        return
    pieces = self.formatted.split('$')
    self.contents = [Constant(pieces[0])]
    for piece in pieces[1:]:
        self.contents += partkey.titlecontents
        self.contents.append(Constant(piece))