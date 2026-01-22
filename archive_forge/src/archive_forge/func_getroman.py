import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def getroman(self):
    """Get the next value as a roman number."""
    result = ''
    number = self.value
    for numeral, value in self.romannumerals:
        if number >= value:
            result += numeral * (number / value)
            number = number % value
    return result