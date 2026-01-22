import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def parsenumbers(self, pos, remaining):
    """Parse the remaining parameters as a running number."""
    'For example, 12 would be {1}{2}.'
    number = self.factory.parsetype(FormulaNumber, pos)
    if not len(number.original) == remaining:
        return number
    for digit in number.original:
        value = self.factory.create(FormulaNumber)
        value.add(FormulaConstant(digit))
        value.type = number
        self.values.append(value)
    return None