import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def processoptions(self):
    """Process all options parsed."""
    if Options.help:
        self.usage()
    if Options.version:
        self.showversion()
    if Options.hardversion:
        self.showhardversion()
    if Options.versiondate:
        self.showversiondate()
    if Options.lyxformat:
        self.showlyxformat()
    if Options.splitpart:
        try:
            Options.splitpart = int(Options.splitpart)
            if Options.splitpart <= 0:
                Trace.error('--splitpart requires a number bigger than zero')
                self.usage()
        except:
            Trace.error('--splitpart needs a numeric argument, not ' + Options.splitpart)
            self.usage()
    if Options.lowmem or Options.toc or Options.tocfor:
        Options.memory = False
    self.parsefootnotes()
    if Options.forceformat and (not Options.imageformat):
        Options.imageformat = Options.forceformat
    if Options.imageformat == 'copy':
        Options.copyimages = True
    if Options.css == []:
        Options.css = ['http://elyxer.nongnu.org/lyx.css']
    if Options.favicon == '':
        pass
    if Options.html:
        Options.simplemath = True
    if Options.toc and (not Options.tocfor):
        Trace.error('Option --toc is deprecated; use --tocfor "page" instead')
        Options.tocfor = Options.toctarget
    if Options.nocopy:
        Trace.error('Option --nocopy is deprecated; it is no longer needed')
    if Options.jsmath:
        Trace.error('Option --jsmath is deprecated; use --mathjax instead')
    for param in dir(Trace):
        if param.endswith('mode'):
            setattr(Trace, param, getattr(self, param[:-4]))