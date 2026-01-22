import pickle
import copy
import functools
import warnings
import pytest
from bs4 import BeautifulSoup
from bs4.element import (
from bs4.builder import (
def test_soupstrainer(self):
    """Parsers should be able to work with SoupStrainers."""
    strainer = SoupStrainer('b')
    soup = self.soup('A <b>bold</b> <meta/> <i>statement</i>', parse_only=strainer)
    assert soup.decode() == '<b>bold</b>'