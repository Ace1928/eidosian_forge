from pdb import set_trace
import pytest
import re
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from bs4.element import (
from . import (
def test_find_all_limit(self):
    """You can limit the number of items returned by find_all."""
    soup = self.soup('<a>1</a><a>2</a><a>3</a><a>4</a><a>5</a>')
    self.assert_selects(soup.find_all('a', limit=3), ['1', '2', '3'])
    self.assert_selects(soup.find_all('a', limit=1), ['1'])
    self.assert_selects(soup.find_all('a', limit=10), ['1', '2', '3', '4', '5'])
    self.assert_selects(soup.find_all('a', limit=0), ['1', '2', '3', '4', '5'])