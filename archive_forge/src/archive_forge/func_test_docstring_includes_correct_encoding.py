import pickle
import copy
import functools
import warnings
import pytest
from bs4 import BeautifulSoup
from bs4.element import (
from bs4.builder import (
def test_docstring_includes_correct_encoding(self):
    soup = self.soup('<root/>')
    assert soup.encode('latin1') == b'<?xml version="1.0" encoding="latin1"?>\n<root/>'