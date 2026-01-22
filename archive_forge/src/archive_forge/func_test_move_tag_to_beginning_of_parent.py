from pdb import set_trace
import pytest
import re
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from bs4.element import (
from . import (
def test_move_tag_to_beginning_of_parent(self):
    data = '<a><b></b><c></c><d></d></a>'
    soup = self.soup(data)
    soup.a.insert(0, soup.d)
    assert '<a><d></d><b></b><c></c></a>' == soup.decode()