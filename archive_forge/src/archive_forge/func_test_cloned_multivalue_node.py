import pytest
import warnings
from bs4 import BeautifulSoup
from bs4.element import SoupStrainer
from . import (
def test_cloned_multivalue_node(self):
    markup = b'<a class="my_class"><p></a>'
    soup = self.soup(markup)
    a1, a2 = soup.find_all('a')
    assert a1 == a2
    assert a1 is not a2