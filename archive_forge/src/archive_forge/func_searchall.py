import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def searchall(self, type):
    """Search for all embedded containers of a given type"""
    list = []
    self.searchprocess(type, lambda container: list.append(container))
    return list