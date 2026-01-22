from bs4.element import (
from . import SoupTest
def test_namespace_may_be_none_or_missing(self):
    a = NamespacedAttribute(None, 'tag')
    assert a == 'tag'
    a = NamespacedAttribute('', 'tag')
    assert a == 'tag'