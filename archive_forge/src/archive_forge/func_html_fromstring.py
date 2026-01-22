from lxml import etree
import sys
import re
import doctest
def html_fromstring(html):
    return etree.fromstring(html, _html_parser)