from pdb import set_trace
import pytest
import re
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from bs4.element import (
from . import (
def test_find_next_for_text_element(self):
    text = self.tree.find(string='One')
    assert text.find_next('b').string == 'Two'
    self.assert_selects(text.find_all_next('b'), ['Two', 'Three'])