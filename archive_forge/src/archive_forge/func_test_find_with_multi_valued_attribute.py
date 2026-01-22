from pdb import set_trace
import pytest
import re
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from bs4.element import (
from . import (
def test_find_with_multi_valued_attribute(self):
    soup = self.soup("<div class='a b'>1</div><div class='a c'>2</div><div class='a d'>3</div>")
    r1 = soup.find('div', 'a d')
    r2 = soup.find('div', re.compile('a d'))
    r3, r4 = soup.find_all('div', ['a b', 'a d'])
    assert '3' == r1.string
    assert '3' == r2.string
    assert '1' == r3.string
    assert '3' == r4.string