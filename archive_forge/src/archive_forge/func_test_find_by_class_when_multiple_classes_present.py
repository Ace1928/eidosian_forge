from pdb import set_trace
import pytest
import re
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from bs4.element import (
from . import (
def test_find_by_class_when_multiple_classes_present(self):
    tree = self.soup("<gar class='foo bar'>Found it</gar>")
    f = tree.find_all('gar', class_=re.compile('o'))
    self.assert_selects(f, ['Found it'])
    f = tree.find_all('gar', class_=re.compile('a'))
    self.assert_selects(f, ['Found it'])
    f = tree.find_all('gar', class_=re.compile('o b'))
    self.assert_selects(f, ['Found it'])