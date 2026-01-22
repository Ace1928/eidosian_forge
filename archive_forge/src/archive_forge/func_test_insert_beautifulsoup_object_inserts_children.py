from pdb import set_trace
import pytest
import re
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from bs4.element import (
from . import (
def test_insert_beautifulsoup_object_inserts_children(self):
    """Inserting one BeautifulSoup object into another actually inserts all
        of its children -- you'll never combine BeautifulSoup objects.
        """
    soup = self.soup("<p>And now, a word:</p><p>And we're back.</p>")
    text = '<p>p2</p><p>p3</p>'
    to_insert = self.soup(text)
    soup.insert(1, to_insert)
    for i in soup.descendants:
        assert not isinstance(i, BeautifulSoup)
    p1, p2, p3, p4 = list(soup.children)
    assert 'And now, a word:' == p1.string
    assert 'p2' == p2.string
    assert 'p3' == p3.string
    assert "And we're back." == p4.string