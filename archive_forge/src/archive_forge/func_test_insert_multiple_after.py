from pdb import set_trace
import pytest
import re
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from bs4.element import (
from . import (
def test_insert_multiple_after(self):
    soup = self.soup('<a>foo</a><b>bar</b>')
    soup.b.insert_after('BAZ', ' ', 'QUUX')
    soup.a.insert_after('QUUX', ' ', 'BAZ')
    assert soup.decode() == self.document_for('<a>foo</a>QUUX BAZ<b>bar</b>BAZ QUUX')
    soup.b.insert_after(soup.a, 'FOO ')
    assert soup.decode() == self.document_for('QUUX BAZ<b>bar</b><a>foo</a>FOO BAZ QUUX')