import pickle
import copy
import functools
import warnings
import pytest
from bs4 import BeautifulSoup
from bs4.element import (
from bs4.builder import (
def test_closing_tag_with_no_opening_tag(self):
    soup = self.soup('<body><div><p>text1</p></span>text2</div></body>')
    assert '<body><div><p>text1</p>text2</div></body>' == soup.body.decode()