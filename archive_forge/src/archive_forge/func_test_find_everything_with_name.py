from pdb import set_trace
import pytest
import re
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from bs4.element import (
from . import (
def test_find_everything_with_name(self):
    """Test an optimization that finds all tags with a given name."""
    soup = self.soup('<a>foo</a><b>bar</b><a>baz</a>')
    assert 2 == len(soup.find_all('a'))