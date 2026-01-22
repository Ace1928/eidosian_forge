from bs4 import BeautifulSoup, NavigableString, Comment, Doctype
from textwrap import fill
import re
import six
def convert_soup(self, soup):
    return self.process_tag(soup, convert_as_inline=False, children_only=True)