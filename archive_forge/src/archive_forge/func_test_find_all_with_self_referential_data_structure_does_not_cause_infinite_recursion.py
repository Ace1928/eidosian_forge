from pdb import set_trace
import pytest
import re
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from bs4.element import (
from . import (
def test_find_all_with_self_referential_data_structure_does_not_cause_infinite_recursion(self):
    soup = self.soup('<a></a>')
    l = []
    l.append(l)
    assert [] == soup.find_all(l)