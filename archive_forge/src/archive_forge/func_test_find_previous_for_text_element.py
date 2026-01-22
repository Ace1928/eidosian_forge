from pdb import set_trace
import pytest
import re
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from bs4.element import (
from . import (
def test_find_previous_for_text_element(self):
    text = self.tree.find(string='Three')
    assert text.find_previous('b').string == 'Three'
    self.assert_selects(text.find_all_previous('b'), ['Three', 'Two', 'One'])