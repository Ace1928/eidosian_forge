from pdb import set_trace
import pytest
import re
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from bs4.element import (
from . import (
def test_find_parent(self):
    assert self.start.find_parent('ul')['id'] == 'bottom'
    assert self.start.find_parent('ul', id='top')['id'] == 'top'