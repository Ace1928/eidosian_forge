from pdb import set_trace
import pytest
import re
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from bs4.element import (
from . import (
def test_find_all_with_non_dictionary_for_attrs_finds_by_class(self):
    soup = self.soup("<a class='bar'>Found it</a>")
    self.assert_selects(soup.find_all('a', re.compile('ba')), ['Found it'])

    def big_attribute_value(value):
        return len(value) > 3
    self.assert_selects(soup.find_all('a', big_attribute_value), [])

    def small_attribute_value(value):
        return len(value) <= 3
    self.assert_selects(soup.find_all('a', small_attribute_value), ['Found it'])