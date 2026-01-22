import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def readequalskey(self, arg, args):
    """Read a key using equals"""
    split = arg.split('=', 1)
    key = split[0]
    value = split[1]
    args.insert(0, value)
    return key