from html.parser import HTMLParser
from itertools import zip_longest
def startswith_whitespace(self):
    return self._leading_whitespace != '' or (self._stripped_data == '' and self._trailing_whitespace != '')