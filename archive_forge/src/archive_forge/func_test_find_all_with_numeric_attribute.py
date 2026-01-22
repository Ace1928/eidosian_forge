from pdb import set_trace
import pytest
import re
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from bs4.element import (
from . import (
def test_find_all_with_numeric_attribute(self):
    tree = self.soup('<a id=1>Unquoted attribute.</a>\n                            <a id="1">Quoted attribute.</a>')
    expected = ['Unquoted attribute.', 'Quoted attribute.']
    self.assert_selects(tree.find_all(id=1), expected)
    self.assert_selects(tree.find_all(id='1'), expected)