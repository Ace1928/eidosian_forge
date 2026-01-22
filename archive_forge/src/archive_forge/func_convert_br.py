from bs4 import BeautifulSoup, NavigableString, Comment, Doctype
from textwrap import fill
import re
import six
def convert_br(self, el, text, convert_as_inline):
    if convert_as_inline:
        return ''
    if self.options['newline_style'].lower() == BACKSLASH:
        return '\\\n'
    else:
        return '  \n'