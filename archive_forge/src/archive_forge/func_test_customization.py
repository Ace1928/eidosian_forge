import warnings
from bs4.element import (
from . import SoupTest
def test_customization(self):
    soup = self.soup('<a class="foo" id="bar">', multi_valued_attributes={'*': 'id'})
    assert soup.a['class'] == 'foo'
    assert soup.a['id'] == ['bar']