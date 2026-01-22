import warnings
from bs4.element import (
from . import SoupTest
def test_accept_charset(self):
    soup = self.soup('<form accept-charset="ISO-8859-1 UTF-8">')
    assert ['ISO-8859-1', 'UTF-8'] == soup.form['accept-charset']