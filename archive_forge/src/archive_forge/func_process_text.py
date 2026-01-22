from bs4 import BeautifulSoup, NavigableString, Comment, Doctype
from textwrap import fill
import re
import six
def process_text(self, el):
    text = six.text_type(el) or ''
    if not (el.parent.name == 'pre' or (el.parent.name == 'code' and el.parent.parent.name == 'pre')):
        text = whitespace_re.sub(' ', text)
    if el.parent.name != 'code' and el.parent.name != 'pre':
        text = self.escape(text)
    if el.parent.name == 'li' and (not el.next_sibling or el.next_sibling.name in ['ul', 'ol']):
        text = text.rstrip()
    return text