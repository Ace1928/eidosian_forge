import warnings
from bs4.element import (
from . import SoupTest
def test_single_value_becomes_list(self):
    soup = self.soup("<a class='foo'>")
    assert ['foo'] == soup.a['class']