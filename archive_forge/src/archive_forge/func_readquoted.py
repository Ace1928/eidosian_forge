import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def readquoted(self, args, initial):
    """Read a value between quotes"""
    Trace.error('Oops')
    value = initial[1:]
    while len(args) > 0 and (not args[0].endswith('"')) and (not args[0].startswith('--')):
        Trace.error('Appending ' + args[0])
        value += ' ' + args[0]
        del args[0]
    if len(args) == 0 or args[0].startswith('--'):
        return None
    value += ' ' + args[0:-1]
    return value