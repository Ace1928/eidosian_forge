import pickle
import copy
import functools
import warnings
import pytest
from bs4 import BeautifulSoup
from bs4.element import (
from bs4.builder import (
def test_non_breaking_spaces_converted_on_the_way_in(self):
    soup = self.soup('<a>&nbsp;&nbsp;</a>')
    assert soup.a.string == '\xa0' * 2