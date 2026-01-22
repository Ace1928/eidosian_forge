import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def searchprocess(self, type, process):
    """Search for elements of a given type and process them"""
    self.locateprocess(lambda container: isinstance(container, type), process)