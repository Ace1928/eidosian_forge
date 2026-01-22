from bs4.element import (
from . import SoupTest
def test_attribute_is_equivalent_to_colon_separated_string(self):
    a = NamespacedAttribute('a', 'b')
    assert 'a:b' == a