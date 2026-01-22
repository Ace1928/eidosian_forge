import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def getparttype(self, type):
    """Obtain the type for the part: without the asterisk, """
    'and switched to Appendix if necessary.'
    if NumberGenerator.appendix and self.getlevel(type) == 1:
        return 'Appendix'
    return self.deasterisk(type)