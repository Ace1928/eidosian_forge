import pytest
from bs4.element import (
from . import SoupTest
def test_doctype_ends_in_newline(self):
    doctype = Doctype('foo')
    soup = self.soup('')
    soup.insert(1, doctype)
    assert soup.encode() == b'<!DOCTYPE foo>\n'