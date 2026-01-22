from bs4.element import (
from . import SoupTest
def test_attributes_are_equivalent_if_prefix_and_name_identical(self):
    a = NamespacedAttribute('a', 'b', 'c')
    b = NamespacedAttribute('a', 'b', 'c')
    assert a == b
    c = NamespacedAttribute('a', 'b', None)
    assert a == c
    d = NamespacedAttribute('a', 'z', 'c')
    assert a != d
    e = NamespacedAttribute('z', 'b', 'c')
    assert a != e