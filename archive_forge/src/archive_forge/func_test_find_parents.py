from pdb import set_trace
import pytest
import re
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from bs4.element import (
from . import (
def test_find_parents(self):
    self.assert_selects_ids(self.start.find_parents('ul'), ['bottom', 'middle', 'top'])
    self.assert_selects_ids(self.start.find_parents('ul', id='middle'), ['middle'])