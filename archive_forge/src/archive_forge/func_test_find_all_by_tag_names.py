from pdb import set_trace
import pytest
import re
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from bs4.element import (
from . import (
def test_find_all_by_tag_names(self):
    self.assert_selects(self.tree.find_all(['a', 'b']), ['First tag.', 'Second tag.', 'Nested tag.'])