import copy
import pickle
import pytest
import sys
from bs4 import BeautifulSoup
from bs4.element import (
from . import (
def test_encoding_substitution_doesnt_happen_if_tag_is_strained(self):
    markup = '<head><meta content="text/html; charset=x-sjis" http-equiv="Content-type"/></head><pre>foo</pre>'
    strainer = SoupStrainer('pre')
    soup = self.soup(markup, parse_only=strainer)
    assert soup.contents[0].name == 'pre'