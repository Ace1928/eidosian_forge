from pdb import set_trace
import pytest
import re
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from bs4.element import (
from . import (
def test_find_everything(self):
    """Test an optimization that finds all tags."""
    soup = self.soup('<a>foo</a><b>bar</b>')
    assert 2 == len(soup.find_all())