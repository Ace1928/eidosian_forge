import pickle
import pytest
import re
import warnings
from . import LXML_PRESENT, LXML_VERSION
from bs4 import (
from bs4.element import Comment, Doctype, SoupStrainer
from . import (
def test_beautifulstonesoup_is_xml_parser(self):
    with warnings.catch_warnings(record=True) as w:
        soup = BeautifulStoneSoup('<b />')
    assert '<b/>' == str(soup.b)
    [warning] = w
    assert warning.filename == __file__
    assert 'BeautifulStoneSoup class is deprecated' in str(warning.message)