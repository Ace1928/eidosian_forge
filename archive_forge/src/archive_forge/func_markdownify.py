from bs4 import BeautifulSoup, NavigableString, Comment, Doctype
from textwrap import fill
import re
import six
def markdownify(html, **options):
    return MarkdownConverter(**options).convert(html)