import pickle
import copy
import functools
import warnings
import pytest
from bs4 import BeautifulSoup
from bs4.element import (
from bs4.builder import (
def test_xml_declaration(self):
    markup = b'<?xml version="1.0" encoding="utf8"?>\n<foo/>'
    soup = self.soup(markup)
    assert markup == soup.encode('utf8')