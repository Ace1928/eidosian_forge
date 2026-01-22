from pdb import set_trace
import pytest
import re
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from bs4.element import (
from . import (
def test_insert_works_on_empty_element_tag(self):
    soup = self.soup('<br/>')
    soup.br.insert(1, 'Contents')
    assert str(soup.br) == '<br>Contents</br>'