from pdb import set_trace
import pytest
import re
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from bs4.element import (
from . import (
def test_find_by_attribute_and_containing_string(self):
    soup = self.soup('<b id="1">foo</b><a id="2">foo</a>')
    a = soup.a
    assert [a] == soup.find_all(id=2, string='foo')
    assert [] == soup.find_all(id=1, string='bar')