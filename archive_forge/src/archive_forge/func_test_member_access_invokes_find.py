import warnings
from bs4.element import (
from . import SoupTest
def test_member_access_invokes_find(self):
    """Accessing a Python member .foo invokes find('foo')"""
    soup = self.soup('<b><i></i></b>')
    assert soup.b == soup.find('b')
    assert soup.b.i == soup.find('b').find('i')
    assert soup.a == None