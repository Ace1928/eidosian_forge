from pdb import set_trace
import pytest
import re
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from bs4.element import (
from . import (
def test_string_set(self):
    """Tag.string = 'string'"""
    soup = self.soup('<a></a> <b><c></c></b>')
    soup.a.string = 'foo'
    assert soup.a.contents == ['foo']
    soup.b.string = 'bar'
    assert soup.b.contents == ['bar']