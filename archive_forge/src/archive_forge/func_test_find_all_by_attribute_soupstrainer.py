from pdb import set_trace
import pytest
import re
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from bs4.element import (
from . import (
def test_find_all_by_attribute_soupstrainer(self):
    tree = self.soup('\n                         <a id="first">Match.</a>\n                         <a id="second">Non-match.</a>')
    strainer = SoupStrainer(attrs={'id': 'first'})
    self.assert_selects(tree.find_all(strainer), ['Match.'])