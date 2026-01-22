import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def parsemultiliner(self, reader, start, ending):
    """Parse a formula in multiple lines"""
    formula = ''
    line = reader.currentline()
    if not start in line:
        Trace.error('Line ' + line.strip() + ' does not contain formula start ' + start)
        return ''
    index = line.index(start)
    line = line[index + len(start):].strip()
    while not line.endswith(ending):
        formula += line + '\n'
        reader.nextline()
        line = reader.currentline()
    formula += line[:-len(ending)]
    reader.nextline()
    return formula