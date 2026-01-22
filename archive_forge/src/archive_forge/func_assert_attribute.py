from pdb import set_trace
import pickle
import pytest
import warnings
from bs4.builder import (
from bs4.builder._htmlparser import BeautifulSoupHTMLParser
from . import SoupTest, HTMLTreeBuilderSmokeTest
def assert_attribute(on_duplicate_attribute, expected):
    soup = self.soup(markup, on_duplicate_attribute=on_duplicate_attribute)
    assert expected == soup.a['href']
    assert ['cls'] == soup.a['class']
    assert 'id' == soup.a['id']