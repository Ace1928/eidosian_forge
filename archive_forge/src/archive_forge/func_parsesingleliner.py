import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def parsesingleliner(self, reader, start, ending):
    """Parse a formula in one line"""
    line = reader.currentline().strip()
    if not start in line:
        Trace.error('Line ' + line + ' does not contain formula start ' + start)
        return ''
    if not line.endswith(ending):
        Trace.error('Formula ' + line + ' does not end with ' + ending)
        return ''
    index = line.index(start)
    rest = line[index + len(start):-len(ending)]
    reader.nextline()
    return rest