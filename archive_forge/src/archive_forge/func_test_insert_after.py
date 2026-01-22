from pdb import set_trace
import pytest
import re
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from bs4.element import (
from . import (
def test_insert_after(self):
    soup = self.soup('<a>foo</a><b>bar</b>')
    soup.b.insert_after('BAZ')
    soup.a.insert_after('QUUX')
    assert soup.decode() == self.document_for('<a>foo</a>QUUX<b>bar</b>BAZ')
    soup.b.insert_after(soup.a)
    assert soup.decode() == self.document_for('QUUX<b>bar</b><a>foo</a>BAZ')
    b = soup.b
    with pytest.raises(ValueError):
        b.insert_after(b)
    b.extract()
    with pytest.raises(ValueError):
        b.insert_after('nope')
    soup = self.soup('<a>')
    soup.a.insert_before(soup.new_tag('a'))