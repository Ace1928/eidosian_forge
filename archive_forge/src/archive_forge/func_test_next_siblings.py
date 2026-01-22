from pdb import set_trace
import pytest
import re
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from bs4.element import (
from . import (
def test_next_siblings(self):
    self.assert_selects_ids(self.start.find_next_siblings('span'), ['2', '3', '4'])
    self.assert_selects_ids(self.start.find_next_siblings(id='3'), ['3'])