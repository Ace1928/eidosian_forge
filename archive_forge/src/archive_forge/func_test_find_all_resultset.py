from pdb import set_trace
import pytest
import re
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from bs4.element import (
from . import (
def test_find_all_resultset(self):
    """All find_all calls return a ResultSet"""
    soup = self.soup('<a></a>')
    result = soup.find_all('a')
    assert hasattr(result, 'source')
    result = soup.find_all(True)
    assert hasattr(result, 'source')
    result = soup.find_all(string='foo')
    assert hasattr(result, 'source')