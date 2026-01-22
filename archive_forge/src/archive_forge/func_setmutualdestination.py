import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def setmutualdestination(self, destination):
    """Set another link as destination, and set its destination to this one."""
    self.destination = destination
    destination.destination = self