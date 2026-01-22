import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def paramdefs(self, readtemplate):
    """Read each param definition in the template"""
    pos = TextPosition(readtemplate)
    while not pos.finished():
        paramdef = ParameterDefinition().parse(pos)
        if paramdef:
            yield paramdef