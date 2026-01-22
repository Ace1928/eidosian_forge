import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def classic(self):
    """Make the contents using classic output generation with XHTML and CSS."""
    whole = FormulaFactory().parseformula(self.parsed)
    FormulaProcessor().process(whole)
    whole.parent = self
    self.contents = [whole]