import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def parsebranch(self, reader):
    """Parse all branch definitions."""
    branch = reader.currentline().split()[1]
    reader.nextline()
    subparser = HeaderParser().complete(HeaderConfig.parameters['endbranch'])
    subparser.parse(reader)
    options = BranchOptions(branch)
    for key in subparser.parameters:
        options.set(key, subparser.parameters[key])
    Options.branches[branch] = options