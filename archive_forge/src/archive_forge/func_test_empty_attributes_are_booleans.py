import pytest
from bs4.element import Tag
from bs4.formatter import (
from . import SoupTest
def test_empty_attributes_are_booleans(self):
    for name in ('html', 'minimal', None):
        formatter = HTMLFormatter.REGISTRY[name]
        assert False == formatter.empty_attributes_are_booleans
    formatter = XMLFormatter.REGISTRY[None]
    assert False == formatter.empty_attributes_are_booleans
    formatter = HTMLFormatter.REGISTRY['html5']
    assert True == formatter.empty_attributes_are_booleans
    formatter = Formatter(empty_attributes_are_booleans=True)
    assert True == formatter.empty_attributes_are_booleans
    for markup in ('<option selected></option>', '<option selected=""></option>'):
        soup = self.soup(markup)
        for formatter in ('html', 'minimal', 'xml', None):
            assert b'<option selected=""></option>' == soup.option.encode(formatter='html')
            assert b'<option selected></option>' == soup.option.encode(formatter='html5')