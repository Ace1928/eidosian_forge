from pdb import set_trace
import pytest
import re
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from bs4.element import (
from . import (
def test_find_tag(self):
    soup = self.soup('<a>1</a><b>2</b><a>3</a><b>4</b>')
    assert soup.find('b').string == '2'