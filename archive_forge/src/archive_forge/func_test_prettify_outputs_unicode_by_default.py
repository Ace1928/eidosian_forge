import copy
import pickle
import pytest
import sys
from bs4 import BeautifulSoup
from bs4.element import (
from . import (
def test_prettify_outputs_unicode_by_default(self):
    soup = self.soup('<a></a>')
    assert str == type(soup.prettify())