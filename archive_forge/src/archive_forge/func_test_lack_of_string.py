import warnings
from bs4.element import (
from . import SoupTest
def test_lack_of_string(self):
    """Only a Tag containing a single text node has a .string."""
    soup = self.soup('<b>f<i>e</i>o</b>')
    assert soup.b.string is None
    soup = self.soup('<b></b>')
    assert soup.b.string is None