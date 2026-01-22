from pdb import set_trace
import pytest
import re
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from bs4.element import (
from . import (
def test_next_of_last_item_is_none(self):
    last = self.tree.find(string='Three')
    assert last.next_element == None