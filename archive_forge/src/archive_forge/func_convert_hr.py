from bs4 import BeautifulSoup, NavigableString, Comment, Doctype
from textwrap import fill
import re
import six
def convert_hr(self, el, text, convert_as_inline):
    return '\n\n---\n\n'