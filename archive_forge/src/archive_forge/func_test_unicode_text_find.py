from pdb import set_trace
import pytest
import re
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from bs4.element import (
from . import (
def test_unicode_text_find(self):
    soup = self.soup('<h1>Räksmörgås</h1>')
    assert soup.find(string='Räksmörgås') == 'Räksmörgås'