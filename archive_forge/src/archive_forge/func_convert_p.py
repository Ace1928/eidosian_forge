from bs4 import BeautifulSoup, NavigableString, Comment, Doctype
from textwrap import fill
import re
import six
def convert_p(self, el, text, convert_as_inline):
    if convert_as_inline:
        return text
    if self.options['wrap']:
        text = fill(text, width=self.options['wrap_width'], break_long_words=False, break_on_hyphens=False)
    return '%s\n\n' % text if text else ''