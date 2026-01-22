import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def readparameters(self, container):
    """Read some size parameters off a container."""
    self.setparameter(container, 'width')
    self.setparameter(container, 'height')
    self.setparameter(container, 'scale')
    self.checkvalidheight(container)
    return self