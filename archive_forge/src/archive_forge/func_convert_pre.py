from bs4 import BeautifulSoup, NavigableString, Comment, Doctype
from textwrap import fill
import re
import six
def convert_pre(self, el, text, convert_as_inline):
    if not text:
        return ''
    code_language = self.options['code_language']
    if self.options['code_language_callback']:
        code_language = self.options['code_language_callback'](el) or code_language
    return '\n```%s\n%s\n```\n' % (code_language, text)