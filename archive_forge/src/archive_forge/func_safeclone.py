import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def safeclone(self, container):
    """Return a new container with contents only in a safe list, recursively."""
    clone = Cloner.clone(container)
    clone.output = container.output
    clone.contents = self.extract(container)
    return clone