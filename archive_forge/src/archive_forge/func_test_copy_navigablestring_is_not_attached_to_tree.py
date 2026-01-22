import copy
import pickle
import pytest
import sys
from bs4 import BeautifulSoup
from bs4.element import (
from . import (
def test_copy_navigablestring_is_not_attached_to_tree(self):
    html = '<b>Foo<a></a></b><b>Bar</b>'
    soup = self.soup(html)
    s1 = soup.find(string='Foo')
    s2 = copy.copy(s1)
    assert s1 == s2
    assert None == s2.parent
    assert None == s2.next_element
    assert None != s1.next_sibling
    assert None == s2.next_sibling
    assert None == s2.previous_element